#include <stdio.h>
#include <assert.h>

#ifdef __CUDACC__
    #define HOST_FUN __host__
    #define DEVICE_FUN __device__
    #define HOST_DEVICE_FUN __host__ __device__
    #include <cuda_runtime.h>
#else
    #define HOST_FUN 
    #define DEVICE_FUN 
    #define HOST_DEVICE_FUN
#endif

void print_cpu_running() {
    printf("No CUDA devices detected. Running on CPU...\n");
}

#ifdef __CUDACC__
    __global__ void print_gpu_running() {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx == 0) {  
            printf("Running GPU computation on thread %d...\n", idx);
        }
    }
#endif

int main(int argc, char **argv) {
    (void)argc, (void)argv;
    #ifdef __CUDACC__
        int threads_per_block = 256;
        int blocks = 1;
        // Launch the kernel
        print_gpu_running<<<blocks, threads_per_block>>>();
        // Check for kernel launch errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
            return 1;
        }
        // Wait for GPU to finish before accessing results
        cudaDeviceSynchronize();
    #else
        print_cpu_running();
    #endif
    return 0;
}

