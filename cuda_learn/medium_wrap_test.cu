#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void warp_heavy_kernel(float *data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        float val = data[i];
        float acc = 0.0f;

        for (int j = 0; j < 10; ++j) { // add computation burden
            float x = sinf(val * 0.001f * (j + 1));
            float y = cosf(val * 0.0001f * (j + 1));
            float z = __expf(x * y + 1e-5f);
            acc += sqrtf(fabsf(z + x - y)) / (1.0f + tid % 33);
        }

        data[i] = acc;
    }
}
__global__ void warp_warmup_kernel(float *data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        float val = data[i];
        float acc = 0.0f;

        for (int j = 0; j < 1000; ++j) { // 增加计算量
            float x = sinf(val * 0.001f * (j + 1));
            float y = cosf(val * 0.0001f * (j + 1));
            float z = __expf(x * y + 1e-5f);
            acc += sqrtf(fabsf(z + x - y)) / (1.0f + tid % 33);
        }

        data[i] = acc;
    }
}
float run_kernel(int blocks, int threads_per_block, float *d_data, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    warp_heavy_kernel<<<blocks, threads_per_block>>>(d_data, N);
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
    const int N = 1 << 20;
    float *h_data = new float[N];
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    printf("Testing warp alignment effects (with warm-up):\n");

    // 预热 GPU
    warp_warmup_kernel<<<2, 1024>>>(d_data, N);
    cudaDeviceSynchronize();

    int configs[][2] = {
        {1, 1024},
        {1, 33},
        {1, 64},
        {1, 128},
        {1, 256},
        {1, 32},
    };

    for (auto &[blocks, threads] : configs) {
        cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        float total_ms = 0.0f;
        int runs = 10;
        for (int i = 0; i < runs; ++i) {
            total_ms += run_kernel(blocks, threads, d_data, N);
        }
        printf("Avg time for %d blocks x %d threads: %.6f ms\n", blocks, threads, total_ms / runs);
    }

    cudaFree(d_data);
    delete[] h_data;
    return 0;
}
