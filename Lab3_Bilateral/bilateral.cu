// bilateral_cuda.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ---------------- CPU version ----------------

static inline int clampInt(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

void bilateralCPU(const unsigned char* input,
                  unsigned char* output,
                  int width,
                  int height,
                  float sigmaD,
                  float sigmaR) {
    const int radius = 1; // 3x3 => 9-point
    const float sigmaD2 = 2.0f * sigmaD * sigmaD;
    const float sigmaR2 = 2.0f * sigmaR * sigmaR;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float center = static_cast<float>(input[y * width + x]);

            float sumWeights = 0.0f;
            float sum = 0.0f;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = clampInt(x + dx, 0, width - 1);
                    int ny = clampInt(y + dy, 0, height - 1);

                    float neighbor = static_cast<float>(input[ny * width + nx]);

                    float spatial = std::exp(-(dx * dx + dy * dy) / sigmaD2);
                    float range = std::exp(-((neighbor - center) * (neighbor - center)) / sigmaR2);

                    float w = spatial * range;
                    sumWeights += w;
                    sum += w * neighbor;
                }
            }

            float value = (sumWeights > 0.0f) ? (sum / sumWeights) : center;
            value = fminf(255.0f, fmaxf(0.0f, value));
            output[y * width + x] = static_cast<unsigned char>(value + 0.5f);
        }
    }
}

// ---------------- GPU version with texture memory ----------------

__global__ void bilateralKernel(cudaTextureObject_t texObj,
                                unsigned char* output,
                                int width,
                                int height,
                                float sigmaD,
                                float sigmaR) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int radius = 1; // 3x3 => 9-point
    const float sigmaD2 = 2.0f * sigmaD * sigmaD;
    const float sigmaR2 = 2.0f * sigmaR * sigmaR;

    float center = tex2D<unsigned char>(texObj, x, y);

    float sumWeights = 0.0f;
    float sum = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float neighbor = tex2D<unsigned char>(texObj, x + dx, y + dy);

            float spatial = expf(-(dx * dx + dy * dy) / sigmaD2);
            float diff = neighbor - center;
            float range = expf(-(diff * diff) / sigmaR2);

            float w = spatial * range;
            sumWeights += w;
            sum += w * neighbor;
        }
    }

    float value = (sumWeights > 0.0f) ? (sum / sumWeights) : center;
    value = fminf(255.0f, fmaxf(0.0f, value));

    output[y * width + x] = static_cast<unsigned char>(value + 0.5f);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <image.bmp> <output.bmp> <sigmaD> <sigmaR>\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    float sigmaD = std::stof(argv[3]);
    float sigmaR = std::stof(argv[4]);

    int width = 0, height = 0, channels = 0;
    unsigned char* img = stbi_load(inputFile.c_str(), &width, &height, &channels, 1);
    if (!img) {
        std::cerr << "Failed to load image: " << inputFile << std::endl;
        return 1;
    }

    size_t bytes = static_cast<size_t>(width) * height * sizeof(unsigned char);

    std::vector<unsigned char> cpuOutput(width * height);
    std::vector<unsigned char> gpuOutput(width * height);

    // CPU timing
    auto cpuStart = std::chrono::high_resolution_clock::now();
    bilateralCPU(img, cpuOutput.data(), width, height, sigmaD, sigmaR);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // CUDA array for texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

    CUDA_CHECK(cudaMemcpy2DToArray(
        cuArray,
        0, 0,
        img,
        width * sizeof(unsigned char),
        width * sizeof(unsigned char),
        height,
        cudaMemcpyHostToDevice
    ));

    // Resource description
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Texture description
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;   // nearest border pixel
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    unsigned char* dOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&dOutput, bytes));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    bilateralKernel<<<grid, block>>>(texObj, dOutput, width, height, sigmaD, sigmaR);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float gpuMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    CUDA_CHECK(cudaMemcpy(gpuOutput.data(), dOutput, bytes, cudaMemcpyDeviceToHost));

    if (!stbi_write_bmp(outputFile.c_str(), width, height, 1, gpuOutput.data())) {
        std::cerr << "Failed to save output BMP: " << outputFile << std::endl;
    }

    std::cout << "Image size: " << width << " x " << height << "\n";
    std::cout << "CPU time: " << cpuMs << " ms\n";
    std::cout << "GPU time: " << gpuMs << " ms\n";
    std::cout << "Output saved to: " << outputFile << "\n";

    // cleanup
    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    CUDA_CHECK(cudaFreeArray(cuArray));
    CUDA_CHECK(cudaFree(dOutput));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    stbi_image_free(img);

    return 0;
}