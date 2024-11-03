#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 10000
#define KERNEL_SIZE 5
// #define DEBUG

int calculate_kernel_sum(int *kernel){
    int kernel_sum = 0;
    for(int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++){
        kernel_sum += kernel[i];
    }
    return kernel_sum;
}

void convolution(int *image, int *kernel, int *output){
    // Note that in this example we assume that the image is square (Nx = Ny = N)
    int kernel_center = (KERNEL_SIZE - 1) / 2;
    int kernel_sum = calculate_kernel_sum(kernel);
    if (kernel_sum == 0) kernel_sum = 1; // Prevent division by zero
    for (int y = 0; y < N; y++){
        for (int x = 0; x < N; x++){
            int sum = 0;
            for (int ky = 0; ky < KERNEL_SIZE; ky++){
                for (int kx = 0; kx < KERNEL_SIZE; kx++){
                    int image_x = x + kx - kernel_center;
                    int image_y = y + ky - kernel_center;
                    if (image_x >= 0 && image_x < N && image_y >= 0 && image_y < N){
                        sum += image[image_y * N + image_x] * kernel[ky * KERNEL_SIZE + kx];
                    }
                }
            }
            output[y * N + x] = sum / kernel_sum;
        }
    }
}

__global__ void convolution_kernel(int *image, int *kernel, int *output, int kernel_sum){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_center = (KERNEL_SIZE - 1) / 2;
    if (x < N && y < N){
        int sum = 0;
        for (int ky = 0; ky < KERNEL_SIZE; ky++){
            for (int kx = 0; kx < KERNEL_SIZE; kx++){
                int image_x = x + kx - kernel_center;
                int image_y = y + ky - kernel_center;
                if (image_x >= 0 && image_x < N && image_y >= 0 && image_y < N){
                    sum += image[image_y * N + image_x] * kernel[ky * KERNEL_SIZE + kx];
                }
            }
        }
        output[y * N + x] = sum / kernel_sum;
    }
}

int main(int argc, char **argv){
    printf("Example of Convolution in CUDA: Using %dx%d Image...\n", N, N);
    // Generate example image of N * N pixels (random values)
    int *image = (int *)malloc(N * N * sizeof(int));
    for(int i = 0; i < N * N; i++){
        image[i] = rand() % 256;
    }
    // Create a kernel of size KERNEL_SIZE * KERNEL_SIZE
    int *kernel = (int *)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(int));
    for(int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++){
        kernel[i] = rand() % 10;
    }
    
    // Convolution Operation on CPU
    clock_t start = clock();
    int *output_cpu = (int *)malloc(N * N * sizeof(int));
    convolution(image, kernel, output_cpu);
    clock_t end = clock();
    printf("Time taken for convolution on CPU: %.4fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Convolution Operation on GPU
    // Allocate memory on the device
    start = clock();
    int *output_gpu = (int *)malloc(N * N * sizeof(int));
    int kernel_sum = calculate_kernel_sum(kernel);
    if (kernel_sum == 0) kernel_sum = 1; // Prevent division by zero
    int *d_image, *d_kernel, *d_output;
    cudaMalloc(&d_image, N * N * sizeof(int));
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(int));
    cudaMalloc(&d_output, N * N * sizeof(int));
    // Copy data to the device
    cudaMemcpy(d_image, image, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output_gpu, N * N * sizeof(int), cudaMemcpyHostToDevice);
    // Launch the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    convolution_kernel<<<numBlocks, threadsPerBlock>>>(d_image, d_kernel, d_output, kernel_sum);
    // Copy the result back to the host
    cudaMemcpy(output_gpu, d_output, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    end = clock();
    printf("Time taken for convolution on GPU: %.4fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    #ifdef DEBUG
    // Print the output of the convolution
    printf("Output of the convolution:\n");
    for (int i = 0; i < 100; i++){
        printf("CPU: %d\t GPU: %d\n", output_cpu[i], output_gpu[i]);        
    }
    #endif

    // Check if the results are the same between CPU and GPU
    for (int i = 0; i < N * N; i++){
        if (output_cpu[i] != output_gpu[i]){
            printf("The results of the convolution are different!\n");
            goto free_memory;
        }
    }
    printf("The results of the convolution are the same!\n");
    
    free_memory:
    printf("Freeing memory...\n");
    free(image);
    free(kernel);
    free(output_cpu);
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return 0;
}
