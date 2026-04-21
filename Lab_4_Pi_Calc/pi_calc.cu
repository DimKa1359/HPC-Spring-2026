#include <iostream>
#include <random>
#include <chrono>
#include <cuda.h>
#include <curand_kernel.h>
#include <cstdlib>

#define BLOCK_SIZE 256

// Ядро 1: генерация точек и частичная редукция
__global__ void pi_kernel(unsigned long long N,
                          unsigned int seed,
                          unsigned long long* block_sums)
{
    __shared__ unsigned int local_counts[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    unsigned int local = 0;
    for (unsigned long long i = idx; i < N; i += gridDim.x * blockDim.x) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y < 1.0f)
            local++;
    }

    local_counts[tid] = local;
    __syncthreads();

    // Редукция внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            local_counts[tid] += local_counts[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        block_sums[blockIdx.x] = local_counts[0];
}

// Ядро 2: финальная редукция по всем блокам
__global__ void reduce_kernel(unsigned long long* block_sums,
                               unsigned long long* result,
                               int num_blocks)
{
    __shared__ unsigned long long shared[BLOCK_SIZE];

    int tid = threadIdx.x;

    unsigned long long sum = 0;
    for (int i = tid; i < num_blocks; i += blockDim.x)
        sum += block_sums[i];

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        *result = shared[0];
}

// Вычисление пи на GPU, время включает memcpy
double compute_pi_gpu(unsigned long long N, double& out_time_sec)
{
    const int blocks = 256;

    unsigned long long* d_block_sums;
    unsigned long long* d_result;

    cudaMalloc(&d_block_sums, blocks * sizeof(unsigned long long));
    cudaMalloc(&d_result, sizeof(unsigned long long));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    pi_kernel<<<blocks, BLOCK_SIZE>>>(N, 1234, d_block_sums);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA ошибка (pi_kernel): " << cudaGetErrorString(err) << "\n";

    reduce_kernel<<<1, BLOCK_SIZE>>>(d_block_sums, d_result, blocks);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA ошибка (reduce_kernel): " << cudaGetErrorString(err) << "\n";

    unsigned long long h_result = 0;
    cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    out_time_sec = ms / 1000.0;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_block_sums);
    cudaFree(d_result);

    return 4.0 * (double)h_result / (double)N;
}

// Вычисление пи на CPU
double compute_pi_cpu(unsigned long long N, double& out_time_sec)
{
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    unsigned long long count = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (unsigned long long i = 0; i < N; i++) {
        float x = dist(gen);
        float y = dist(gen);
        if (x * x + y * y < 1.0f)
            count++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    out_time_sec = std::chrono::duration<double>(t_end - t_start).count();

    return 4.0 * (double)count / (double)N;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Использование: pi_calc <N>\n";
        return 1;
    }

    unsigned long long N = strtoull(argv[1], nullptr, 10);

    double cpu_time = 0.0, gpu_time = 0.0;

    double pi_cpu = compute_pi_cpu(N, cpu_time);
    double pi_gpu = compute_pi_gpu(N, gpu_time);

    std::cout << "CPU time: " << cpu_time << " s\n";
    std::cout << "Pi (CPU): " << pi_cpu   << "\n\n";
    std::cout << "GPU time: " << gpu_time << " s\n";
    std::cout << "Pi (GPU): " << pi_gpu   << "\n";

    return 0;
}