#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_from_gpu()
{
    printf("Hello from the GPU\n");
}

int main(void)
{
    // Launch kernel with 4 blocks, 4 threads per block
    hello_from_gpu<<<2, 2>>>();

    // Ensure the device completes execution before exiting
    cudaDeviceSynchronize();

    return 0;
}
