#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Blocks Per Grid: (%d, %d, %d)\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
    printf("Warp Size: %d\n", prop.warpSize);
    
    return 0;
}
