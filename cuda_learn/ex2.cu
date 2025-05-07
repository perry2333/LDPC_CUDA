#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_from_gpu()
{
    //dim3 grid_size(2,2);
    //dim3 block_size(4,4);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from block %d and thread %d, global id %d\n", bid, tid, id);
}

int main(void)
{
    // Launch kernel with 4 blocks, 4 threads per block
    hello_from_gpu<<<2, 8>>>();

    // Ensure the device completes execution before exiting
    cudaDeviceSynchronize();

    return 0;
}
