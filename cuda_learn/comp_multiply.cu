#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "time_meas.h"  // Ensure this header is available for CPU timing

// Scalar version from original code
void componentwise_multiply_real_scalar(int16_t *x, int16_t *y, int16_t *z, uint16_t N) {
    for (int i = 0; i < N; i++) {
        z[i] = (((int32_t)x[i]) * y[i]) >> 15;
    }
}

// CUDA kernel
__global__ void componentwise_multiply_real_cuda(int16_t *x, int16_t *y, int16_t *z, uint16_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int32_t product = (int32_t)x[i] * (int32_t)y[i];
        z[i] = (int16_t)(product >> 15);
    }
}

#define LENGTH 10000

int main() {
    // Allocate and initialize host memory with alignment
    int16_t x[LENGTH] __attribute__((aligned(32))) = {0};
    int16_t y[LENGTH] __attribute__((aligned(32))) = {0};
    int16_t z[LENGTH] __attribute__((aligned(32))) = {0};

    for (int i = 0; i < LENGTH; i++) {
        x[i] = i * 255;
        y[i] = i * 13;
    }

    // Time measurement variables for CPU
    time_stats_t time_meas;

    // Time scalar version on CPU
    reset_meas(&time_meas);
    for (int trial = 0; trial < 10000; trial++) {
        start_meas(&time_meas);
        componentwise_multiply_real_scalar(x, y, z, LENGTH);
        stop_meas(&time_meas);
    }
    printf("Scalar avg time %d (cycles) max %d \n", (int)(time_meas.diff / time_meas.trials), (int)time_meas.max);

    // CUDA part
    
    int16_t *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, LENGTH * sizeof(int16_t));
    cudaMalloc((void**)&d_y, LENGTH * sizeof(int16_t));
    cudaMalloc((void**)&d_z, LENGTH * sizeof(int16_t));

    // Copy data from host to device
    cudaMemcpy(d_x, x, LENGTH * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, LENGTH * sizeof(int16_t), cudaMemcpyHostToDevice);

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (LENGTH + threadsPerBlock - 1) / threadsPerBlock;

    // Warm-up kernel to avoid initialization overhead
    componentwise_multiply_real_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, LENGTH);
    cudaDeviceSynchronize();


    // Time the kernel over 10000 trials
    cudaEventRecord(start);
    reset_meas(&time_meas);
    
    for (int trial = 0; trial < 10000; trial++) {
        start_meas(&time_meas);
        componentwise_multiply_real_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, LENGTH);
        stop_meas(&time_meas);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    printf("(CUDA)Scalar avg time %d (cycles) max %d \n", (int)(time_meas.diff / time_meas.trials), (int)time_meas.max);
    // Calculate and print time
    float totalTimeMs;
    cudaEventElapsedTime(&totalTimeMs, start, stop);
    float avgTimeMs = totalTimeMs / 10000;
    printf("CUDA avg time: %f ms\n", avgTimeMs);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}