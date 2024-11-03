#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "utils.h"

#ifdef __CUDACC__
    #define HOST_FUN __host__
    #define DEVICE_FUN __device__
    #define HOST_DEVICE_FUN __host__ __device__
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
#else
    #define HOST_FUN 
    #define DEVICE_FUN 
    #define HOST_DEVICE_FUN
#endif

void print_cpu_running() {
    printf("No CUDA devices detected. Running on CPU...\n");
}

#ifdef __CUDACC__
    void print_gpu_device_check() {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (deviceCount > 0 && err == cudaSuccess) {
            printf("CUDA device detected. Running on GPU...\n");
        } else {
            printf("No CUDA devices detected. Running on CPU...\n");
        }
    }
#endif

float dotProduct(float *weights, int *x, int features) {
    float result = 0;
    for (int i = 0; i < features; i++) {
        result += weights[i] * x[i];
    }
    return result;
}

void trainPegasosSVM(float *weights, Dataset trainData, float lambda, int iterations) {
    for (int step = 1; step < (iterations + 1); step++) {
        #ifdef DEBUG  
            // Print the first weight every 1000 iterations
            if (step % 1000 == 0) {
                printf("Weight 0 at Iteration: %d, Weight: %f\n", step, weights[0]);
            }
        #endif
        int randomIndex = rand() % trainData.instances;
        int *x = trainData.input[randomIndex];
        int y = trainData.output[randomIndex];
        float step_size = 1.0 / (lambda * step);
        if (y * dotProduct(weights, x, trainData.features) < 1) {
            for (int i = 0; i < trainData.features; i++) {
                weights[i] = (1 - step_size * lambda) * weights[i] + step_size * y * x[i];
            }
        } else if (y * dotProduct(weights, x, trainData.features) >= 1) {
            for (int i = 0; i < trainData.features; i++) {
                weights[i] = (1 - step_size * lambda) * weights[i];
            }
        }
        // Gradient-Projection step
        float ball_radius = 1.0 / sqrt(lambda);
        float norm = 0;
        for (int i = 0; i < trainData.features; i++) {
            norm += weights[i] * weights[i];
        }
        float scaling_factor = fmin(1.0, ball_radius / sqrt(norm));
        for (int i = 0; i < trainData.features; i++) {
            weights[i] *= scaling_factor;
        }
    }
}

#ifdef __CUDACC__
    __global__ void trainBatchedPegasosSVMKernel(float *weights, int *d_input, int *d_output, int num_instances, int num_features, float lambda, int iterations, int batch_size) {
        extern __shared__ float shared_weights[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Initialize shared weights
        if (tid < num_features) {
            shared_weights[tid] = weights[tid];
        }
        __syncthreads();

        // Initialize random state
        curandState state;
        curand_init(clock64(), idx, 0, &state);

        for (int step = 1; step < (iterations + 1); step++) {
            float step_size = 1.0 / (lambda * step);

            // Sample batch indices
            __shared__ int batch_indices[256];
            if (tid < batch_size) {
                batch_indices[tid] = curand(&state) % num_instances;
            }
            __syncthreads();

            // Compute gradient
            float gradient = 0.0f;
            for (int i = 0; i < batch_size; i++) {
                int data_idx = batch_indices[i];
                int label = d_output[data_idx];
                int feature_offset = data_idx * num_features + tid;
                float dot_product = 0.0f;

                // Compute dot product
                for (int j = 0; j < num_features; j++) {
                    dot_product += shared_weights[j] * d_input[data_idx * num_features + j];
                }

                // Update gradient if constraint is violated
                if (label * dot_product < 1.0f) {
                    gradient += label * d_input[feature_offset];
                }
            }

            // Update weights
            if (tid < num_features) {
                shared_weights[tid] = (1.0 - step_size * lambda) * shared_weights[tid] + (step_size / batch_size) * gradient;
            }
            __syncthreads();

            // Projection step
            if (tid == 0) {
                float norm = 0.0;
                for (int i = 0; i < num_features; i++) {
                    norm += shared_weights[i] * shared_weights[i];
                }
                float scale = fminf(1.0, (1.0 / sqrtf(lambda)) / sqrtf(norm));
                for (int i = 0; i < num_features; i++) {
                    shared_weights[i] *= scale;
                }
            }
            __syncthreads();
        }

        // Write back updated weights
        if (tid < num_features) {
            weights[tid] = shared_weights[tid];
        }
    }
#endif

void trainBatchedPegasosSVM(float *weights, Dataset trainData, float lambda, int iterations, int batch_size) {
    for (int step = 1; step < (iterations + 1); step++) {
        #ifdef DEBUG
            // Print the first weight every 1000 iterations
            if (step % 1000 == 0) {
                printf("Weight 0 at Iteration: %d, Weight: %f\n", step, weights[0]);
            }
        #endif
        // Select batch_size random instances
        int *batch_indices = (int *)malloc(batch_size * sizeof(int));
        for (int i = 0; i < batch_size; i++) {
            batch_indices[i] = rand() % trainData.instances;
        }
        float step_size = 1.0 / (lambda * step);
        float *sum_positive_instances = (float *)calloc(trainData.features, sizeof(float));
        for (int i = 0; i < batch_size; i++) {
            int *x = trainData.input[batch_indices[i]];
            int y = trainData.output[batch_indices[i]];
            if (y * dotProduct(weights, x, trainData.features) < 1) {
                for (int j = 0; j < trainData.features; j++) {
                    sum_positive_instances[j] += y * x[j];
                }
            }
        }
        for (int i = 0; i < trainData.features; i++) {
            weights[i] = (1 - step_size * lambda) * weights[i] + (step_size / batch_size) * sum_positive_instances[i];
        }
        // Gradient-Projection step
        float ball_radius = 1.0 / sqrt(lambda);
        float norm = 0;
        for (int i = 0; i < trainData.features; i++) {
            norm += weights[i] * weights[i];
        }
        float scaling_factor = fmin(1.0, ball_radius / sqrt(norm));
        for (int i = 0; i < trainData.features; i++) {
            weights[i] *= scaling_factor;
        }
    }
}

void predictPegasosSVM(int *predictions, float *weights, Dataset testData) {
    for (int inst = 0; inst < testData.instances; inst++) {
        int *x = testData.input[inst];
        float result = dotProduct(weights, x, testData.features);
        predictions[inst] = result > 0 ? 1 : -1;
    }
}

const static char *TRAIN_DATA;
const static char *TEST_DATA;

int main(int argc, char **argv) {
    (void)argc, (void)argv;
    // Print the running device
    #ifdef __CUDACC__
        print_gpu_device_check();
    #else
        print_cpu_running();
    #endif

    // Check if argument is mush or rcv1 and load the different datasets
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <dataset>\n", argv[0]);
        return 1;
    } 

    if (argv[1] == NULL) {
        fprintf(stderr, "Dataset not found. Please use either 'mush' or 'rcv1'\n");
        return 1;
    }

    if (strcmp(argv[1], "mush") == 0) {
        TRAIN_DATA = "data/mush_train.data";
        TEST_DATA = "data/mush_test.data";

        // Load the training and testing datasets
        Dataset trainData = readDataset(TRAIN_DATA, TRAIN);
        Dataset testData = readDataset(TEST_DATA, TEST);
        assert(trainData.features == testData.features && "Number of features in the training and testing datasets should be the same");
        int features = trainData.features;
        printf("Number of features: %d\n", features);
        printf("\nMethod 1: Stochastic Pegasos SVM\n");
        clock_t start = clock();
        // Initialize the weights to zero
        float *weights = (float *)calloc(features, sizeof(float));
        // Initalize the regularization lambda param to 2*10^-4
        float lambda = 2e-4;
        // Number of iterations for the Pegasos Algorithm
        int iterations = 10000;
        // Train the SVM model using the Pegasos Algorithm
        trainPegasosSVM(weights, trainData, lambda, iterations);
        // Make predictions
        int *predictions = (int *)malloc(testData.instances * sizeof(int));
        predictPegasosSVM(predictions, weights, testData);
        // Evaluate the predictions
        int correct = 0;
        for (int i = 0; i < testData.instances; i++) {
            #ifdef DEBUG
                printf("Prediction: %d, Actual: %d\n", predictions[i], testData.output[i]);
            #endif
            if (predictions[i] == testData.output[i]) {
                correct++;
            }
        }
        float accuracy = (float)correct / testData.instances;
        clock_t end = clock();
        printf("Accuracy: %.2f\n", accuracy);
        printf("Time taken: %.4f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        free(weights);
        free(predictions);

        printf("\nMethod 2: Mini-Batch Pegasos SVM\n");
        start = clock();
        // Initialize the weights to zero
        weights = (float *)calloc(features, sizeof(float));
        // Regularization lambda parameter
        lambda = 2e-4;
        // Number of iterations
        iterations = 10000;
        // Batch size for the Mini-Batch Pegasos Algorithm
        int batch_size = 124;
        // Train the SVM model using the Pegasos Algorithm
        trainBatchedPegasosSVM(weights, trainData, lambda, iterations, batch_size);
        // Make predictions
        predictions = (int *)malloc(testData.instances * sizeof(int));
        predictPegasosSVM(predictions, weights, testData);
        // Evaluate the predictions
        correct = 0;
        for (int i = 0; i < testData.instances; i++) {
            #ifdef DEBUG
                printf("Prediction: %d, Actual: %d\n", predictions[i], testData.output[i]);
            #endif
            if (predictions[i] == testData.output[i]) {
                correct++;
            }
        }
        accuracy = (float)correct / testData.instances;
        end = clock();
        printf("Accuracy: %.2f\n", accuracy);
        printf("Time taken: %.4f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
        free(weights);
        free(predictions);

        #ifdef __CUDACC__
            printf("\nMethod 3: Mini-Batch Pegasos SVM (CUDA-Accelerated Version)\n");
            // Initialize the weights to zero
            weights = (float *)calloc(features, sizeof(float));
            // Regularization lambda parameter
            lambda = 2e-4;
            // Number of iterations
            iterations = 10000;
            // Batch size for the Mini-Batch Pegasos Algorithm
            batch_size = 256;
            int threadsPerBlock = 256;
            int numBlocks = (features + threadsPerBlock - 1) / threadsPerBlock;
            // Allocate device memory for weights, input, and output
            float *d_weights;
            int *d_input;
            int *d_output;
            cudaMalloc(&d_weights, features * sizeof(float));
            cudaMalloc(&d_input, trainData.instances * features * sizeof(int));
            cudaMalloc(&d_output, trainData.instances * sizeof(int));
            // Copy weights, input, and output to device
            cudaMemcpy(d_weights, weights, features * sizeof(float), cudaMemcpyHostToDevice);
            // Flatten input data
            int *h_input = (int *)malloc(trainData.instances * features * sizeof(int));
            for (int i = 0; i < trainData.instances; i++) {
                memcpy(&h_input[i * features], trainData.input[i], features * sizeof(int));
            }
            cudaMemcpy(d_input, h_input, trainData.instances * features * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_output, trainData.output, trainData.instances * sizeof(int), cudaMemcpyHostToDevice);
            // Train the SVM model using the Pegasos Algorithm
            trainBatchedPegasosSVMKernel<<<numBlocks, threadsPerBlock>>>(d_weights, d_input, d_output, trainData.instances, features, lambda, iterations, batch_size);
            // Copy back the weights
            cudaMemcpy(weights, d_weights, features * sizeof(float), cudaMemcpyDeviceToHost);
            // Free device memory
            cudaFree(d_weights);
            cudaFree(d_input);
            cudaFree(d_output);
            free(h_input);
            // Make predictions
            predictions = (int *)malloc(testData.instances * sizeof(int));
            predictPegasosSVM(predictions, weights, testData);
            // Evaluate the predictions
            correct = 0;
            for (int i = 0; i < testData.instances; i++) {
                // printf("Prediction: %d, Actual: %d\n", predictions[i], testData.output[i]);
                if (predictions[i] == testData.output[i]) {
                    correct++;
                }
            }
            accuracy = (float)correct / testData.instances;
            end = clock();
            printf("Accuracy: %.2f\n", accuracy);
            printf("Time taken: %.4f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
            free(weights);
            free(predictions);
        #endif
        // Free the allocated memory
        freeDataset(trainData);
        freeDataset(testData);

    } else if (strcmp(argv[1], "rcv1") == 0) {
        TRAIN_DATA = "data/rcv1_train.data";
        TEST_DATA = "data/rcv1_test.data";
        fprintf(stderr, "RCV1 dataset not implemented yet\n");
        return -1;
    } else {
        fprintf(stderr, "Dataset not found. Please use either 'mush' or 'rcv1'\n");
        return 1;
    }

    return 0;
}
