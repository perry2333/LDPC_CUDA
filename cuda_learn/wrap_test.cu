#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

typedef float real;

__global__ void heavy_compute_kernel(float* data, const float target, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = data[idx];
    float accum = 0.0f;

    // 让每个线程进行大量复杂运算，并强依赖其 thread ID
    for (int i = 0; i < 10000; ++i) {
        float x = sinf(val + i * 0.001f);
        float y = cosf(val + i * 0.0005f);
        accum += sqrtf(fabsf(x * y)) + threadIdx.x * 0.0001f;
    }

    data[idx] = accum;
}

// Kernel 执行浮点运算模拟计算负载
__global__ void arithmetic(real *d_x, const real x0, const int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        real x = d_x[idx];
        while (sqrtf(x) < x0*10000) {
            x += 1.0f;
        }
        d_x[idx] = x;
    }
}

// 定时并运行 kernel 的函数
float run_kernel(int threads_per_block, int num_blocks, real *d_x, real x0, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    heavy_compute_kernel<<<num_blocks, threads_per_block>>>(d_x, x0, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    const real x0 = 100.0f;

    // 测试用配置（固定 2 blocks，改变 thread 数）
    const int num_configs = 7;
    int thread_counts[num_configs] = {32, 33, 40, 64, 96, 1000, 1024};

    // GPU 预热（提前执行一次 kernel）
    int warmup_N = 64;
    real *warmup_x;
    cudaMalloc(&warmup_x, warmup_N * sizeof(real));
    cudaMemset(warmup_x, 0, warmup_N * sizeof(real));
    run_kernel(32, 2, warmup_x, x0, warmup_N);
    cudaFree(warmup_x);

    printf("Testing warp alignment effects:\n");
    for (int i = 0; i < num_configs; ++i) {
        int threads_per_block = thread_counts[i];
        int num_blocks = 2;
        int N = threads_per_block * num_blocks;

        real *h_x = new real[N];
        for (int j = 0; j < N; ++j) h_x[j] = 1.0f;

        real *d_x;
        cudaMalloc(&d_x, N * sizeof(real));
        cudaMemcpy(d_x, h_x, N * sizeof(real), cudaMemcpyHostToDevice);

        float time_ms = run_kernel(threads_per_block, num_blocks, d_x, x0, N);

        printf("Time for %d blocks x %d threads: %.6f ms\n", num_blocks, threads_per_block, time_ms);

        cudaFree(d_x);
        delete[] h_x;
    }

    return 0;
}
