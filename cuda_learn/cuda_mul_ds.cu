#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <chrono>  // 高精度计时库

// 删除原time_meas相关代码

// Scalar版本保持不变
void componentwise_multiply_real_scalar(int16_t *x, int16_t *y, int16_t *z, uint16_t N) {
    for (int i = 0; i < N; i++) {
        z[i] = (((int32_t)x[i]) * y[i]) >> 15;
    }
}

// CUDA kernel保持不变
__global__ void componentwise_multiply_real_cuda(int16_t *x, int16_t *y, int16_t *z, uint16_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int32_t product = (int32_t)x[i] * (int32_t)y[i];
        z[i] = (int16_t)(product >> 15);
    }
}

#define LENGTH 10000

int main() {
    // 初始化数据保持不变
    int16_t x[LENGTH] __attribute__((aligned(32))) = {0};
    int16_t y[LENGTH] __attribute__((aligned(32))) = {0};
    int16_t z[LENGTH] __attribute__((aligned(32))) = {0};

    for (int i = 0; i < LENGTH; i++) {
        x[i] = i * 255;
        y[i] = i * 13;
    }

    // 新CPU计时方法（单位：毫秒）
    // --------------------------------------------------
    double total_cpu_time = 0.0;
    for (int trial = 0; trial < 10000; trial++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        componentwise_multiply_real_scalar(x, y, z, LENGTH);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        total_cpu_time += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    }
    printf("CPU avg time: %.6f ms\n", total_cpu_time / 10000);
    // --------------------------------------------------

    // CUDA部分（保留事件计时并改进输出）
    // --------------------------------------------------
    int16_t *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, LENGTH * sizeof(int16_t));
    cudaMalloc(&d_y, LENGTH * sizeof(int16_t));
    cudaMalloc(&d_z, LENGTH * sizeof(int16_t));

    cudaMemcpy(d_x, x, LENGTH * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, LENGTH * sizeof(int16_t), cudaMemcpyHostToDevice);

    // 配置CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_gpu_time = 0.0f;

    // 预热
    componentwise_multiply_real_cuda<<<(LENGTH+255)/256, 256>>>(d_x, d_y, d_z, LENGTH);
    cudaDeviceSynchronize();

    // 精确测量
    for (int trial = 0; trial < 10000; trial++) {
        cudaEventRecord(start);
        componentwise_multiply_real_cuda<<<(LENGTH+255)/256, 256>>>(d_x, d_y, d_z, LENGTH);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  // 每次执行后同步确保精确计时
        
        float trial_time;
        cudaEventElapsedTime(&trial_time, start, stop);
        total_gpu_time += trial_time;
    }
    printf("GPU avg time: %.6f ms\n", total_gpu_time / 10000);
    // --------------------------------------------------

    // 清理资源保持不变
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}