#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "time_meas.h"  // Ensure this header is available for CPU timing

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(__rdtsc)
#else
#include <x86intrin.h>
#endif

// 使用 sleep(1) 测量 CPU 频率，单位为Hz
uint64_t measure_cpu_freq() {
    struct timespec start_time, end_time;
    uint64_t start_cycles, end_cycles;
    
    // 获取初始时间和TSC计数
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    start_cycles = __rdtsc();
    
    // 延迟1秒
    sleep(1);
    
    // 结束时刻的时间和TSC计数
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    end_cycles = __rdtsc();
    
    // 计算精确经过的秒数
    double elapsed = (end_time.tv_sec - start_time.tv_sec) +
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    // CPU频率：经过的TSC周期数除以实际秒数
    return (uint64_t)((end_cycles - start_cycles) / elapsed);
}

// Scalar version from original code
void componentwise_multiply_real_scalar(int16_t *x, int16_t *y, int16_t *z, uint16_t N) {
    for (int i = 0; i < N; i++) {
        z[i] = (((int16_t)x[i]) * y[i]) >> 15;
    }
}

// CUDA kernel
__global__ void componentwise_multiply_real_cuda(int16_t *x, int16_t *y, int16_t *z, uint16_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int32_t product = (int16_t)x[i] * (int16_t)y[i];
        z[i] = (int16_t)(product >> 15);
    }
}

#define LENGTH 10000

int main() {
    // 动态获取CPU频率
    uint64_t cpu_freq = measure_cpu_freq();
    printf("Measured CPU frequency: %llu Hz\n", (unsigned long long)cpu_freq);
    
    // 分配并初始化内存（注意内存对齐）
    int16_t x[LENGTH] __attribute__((aligned(32))) = {0};
    int16_t y[LENGTH] __attribute__((aligned(32))) = {0};
    int16_t z[LENGTH] __attribute__((aligned(32))) = {0};

    for (int i = 0; i < LENGTH; i++) {
        x[i] = i * 255;
        y[i] = i * 13;
    }

    // CPU计时变量
    time_stats_t time_meas;

    // CPU上的标量版本计时（10000次试验）
    reset_meas(&time_meas);
    for (int trial = 0; trial < 10000; trial++) {
        start_meas(&time_meas);
        componentwise_multiply_real_scalar(x, y, z, LENGTH);
        stop_meas(&time_meas);
    }
    // 计算CPU平均时钟周期数，并转换为毫秒
    uint64_t avg_cycles = time_meas.diff / time_meas.trials;
    double avg_cpu_ms = ((double)avg_cycles / cpu_freq) * 1000;
    printf("Scalar CPU avg time: %llu cycles, which is %f ms; max cycles %llu\n",
           (unsigned long long)avg_cycles, avg_cpu_ms, (unsigned long long)time_meas.max);

    // CUDA部分
    int16_t *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, LENGTH * sizeof(int16_t));
    cudaMalloc((void**)&d_y, LENGTH * sizeof(int16_t));
    cudaMalloc((void**)&d_z, LENGTH * sizeof(int16_t));

    // 从主机拷贝数据到设备
    cudaMemcpy(d_x, x, LENGTH * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, LENGTH * sizeof(int16_t), cudaMemcpyHostToDevice);

    // 设置CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 设置CUDA核函数配置
    int threadsPerBlock = 256;
    int blocksPerGrid = (LENGTH + threadsPerBlock - 1) / threadsPerBlock;

    // 预热核函数以消除初始化延迟
    componentwise_multiply_real_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, LENGTH);
    cudaDeviceSynchronize();

    // 使用CUDA事件计时10000次内核调用
    cudaEventRecord(start);
    for (int trial = 0; trial < 10000; trial++) {
        componentwise_multiply_real_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, LENGTH);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalTimeMs;
    cudaEventElapsedTime(&totalTimeMs, start, stop);
    float avg_gpu_ms = totalTimeMs / 10000;
    printf("CUDA avg time: %f ms\n", avg_gpu_ms);

    // 清理资源
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
