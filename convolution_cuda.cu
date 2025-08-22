/**
 * 文件名: convolution_cuda.cu
 * 描述: 使用CUDA实现的卷积计算
 * 
 * 卷积原理:
 * 卷积是信号处理和图像处理中的基本操作。在离散情况下，它表示为：
 * (f * g)[n] = Σ f[m] * g[n-m]
 * 
 * 在图像处理中，二维卷积公式为:
 * O[i,j] = Σ Σ I[i+m,j+n] * K[m,n]
 * 
 * CUDA并行化思路:
 * 1. 每个CUDA线程负责计算输出矩阵的一个元素
 * 2. 利用GPU的大规模并行计算能力加速卷积运算
 * 3. 对共享内存和全局内存访问进行优化
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>

// CUDA错误检查宏
#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s, at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }

// 设备端卷积计算核函数 - 基础版本
__global__ void convolutionKernel(float* input, float* kernel, float* output, 
                                  int inputRows, int inputCols, 
                                  int kernelRows, int kernelCols,
                                  int outputRows, int outputCols) {
    // 计算当前线程负责的输出矩阵位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确保在输出矩阵范围内
    if (row < outputRows && col < outputCols) {
        float sum = 0.0f;
        
        // 应用卷积核
        for (int ki = 0; ki < kernelRows; ki++) {
            for (int kj = 0; kj < kernelCols; kj++) {
                int inputRow = row + ki;
                int inputCol = col + kj;
                sum += input[inputRow * inputCols + inputCol] * kernel[ki * kernelCols + kj];
            }
        }
        
        // 写入结果
        output[row * outputCols + col] = sum;
    }
}

// 设备端卷积计算核函数 - 使用共享内存的优化版本
__global__ void convolutionKernelShared(float* input, float* kernel, float* output, 
                                        int inputRows, int inputCols, 
                                        int kernelRows, int kernelCols,
                                        int outputRows, int outputCols) {
    extern __shared__ float sharedKernel[];
    
    // 计算当前线程负责的输出矩阵位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 将卷积核加载到共享内存中 - 所有线程协作完成
    if (threadIdx.y < kernelRows && threadIdx.x < kernelCols) {
        sharedKernel[threadIdx.y * kernelCols + threadIdx.x] = kernel[threadIdx.y * kernelCols + threadIdx.x];
    }
    
    // 确保所有线程都加载完成
    __syncthreads();
    
    // 确保在输出矩阵范围内
    if (row < outputRows && col < outputCols) {
        float sum = 0.0f;
        
        // 应用卷积核 - 从共享内存读取卷积核数据
        for (int ki = 0; ki < kernelRows; ki++) {
            for (int kj = 0; kj < kernelCols; kj++) {
                int inputRow = row + ki;
                int inputCol = col + kj;
                sum += input[inputRow * inputCols + inputCol] * sharedKernel[ki * kernelCols + kj];
            }
        }
        
        // 写入结果
        output[row * outputCols + col] = sum;
    }
}

// 为矩阵分配CPU内存
float* allocate_matrix_cpu(int rows, int cols) {
    float* matrix = (float*)malloc(rows * cols * sizeof(float));
    if (!matrix) {
        fprintf(stderr, "CPU memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return matrix;
}

// 为矩阵分配GPU内存
float* allocate_matrix_gpu(int rows, int cols) {
    float* dev_matrix;
    cudaError_t err = cudaMalloc((void**)&dev_matrix, rows * cols * sizeof(float));
    CUDA_CHECK_ERROR(err);
    return dev_matrix;
}

// 创建随机矩阵
void create_random_matrix(float* matrix, int rows, int cols, float min_val, float max_val) {
    // 初始化随机数生成器(仅初始化一次，避免快速重复调用导致相同序列)
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 生成范围在[min_val, max_val]的随机浮点数
            float random_val = min_val + ((float)rand() / RAND_MAX) * (max_val - min_val);
            matrix[i * cols + j] = random_val;
        }
    }
}

// 打印矩阵
void print_matrix(float* matrix, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 使用CPU执行卷积计算(用于结果比较)
void convolve_cpu(float* input, float* kernel, float* output,
                 int inputRows, int inputCols,
                 int kernelRows, int kernelCols) {
    // 计算输出矩阵的尺寸
    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;
    
    // 计算卷积
    for (int i = 0; i < outputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
            // 对当前位置应用卷积核
            float sum = 0.0f;
            for (int ki = 0; ki < kernelRows; ki++) {
                for (int kj = 0; kj < kernelCols; kj++) {
                    sum += input[(i + ki) * inputCols + (j + kj)] * kernel[ki * kernelCols + kj];
                }
            }
            output[i * outputCols + j] = sum;
        }
    }
}

// 使用CUDA执行卷积计算
void convolve_cuda(float* h_input, float* h_kernel, float* h_output,
                  int inputRows, int inputCols,
                  int kernelRows, int kernelCols,
                  bool useSharedMemory = false) {
    // 计算输出矩阵的尺寸
    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;
    
    // 分配设备内存
    float* d_input = allocate_matrix_gpu(inputRows, inputCols);
    float* d_kernel = allocate_matrix_gpu(kernelRows, kernelCols);
    float* d_output = allocate_matrix_gpu(outputRows, outputCols);
    
    // 将输入数据从主机内存复制到设备内存
    cudaError_t err = cudaMemcpy(d_input, h_input, inputRows * inputCols * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(err);
    
    err = cudaMemcpy(d_kernel, h_kernel, kernelRows * kernelCols * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(err);
    
    // 定义CUDA线程块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((outputCols + blockSize.x - 1) / blockSize.x, 
                  (outputRows + blockSize.y - 1) / blockSize.y);
    
    // 根据参数选择使用基础版本还是共享内存优化版本
    if (useSharedMemory) {
        // 计算共享内存大小 - 用于存储卷积核
        int sharedMemSize = kernelRows * kernelCols * sizeof(float);
        
        // 启动CUDA核函数 - 共享内存版本
        convolutionKernelShared<<<gridSize, blockSize, sharedMemSize>>>(
            d_input, d_kernel, d_output,
            inputRows, inputCols,
            kernelRows, kernelCols,
            outputRows, outputCols
        );
    } else {
        // 启动CUDA核函数 - 基础版本
        convolutionKernel<<<gridSize, blockSize>>>(
            d_input, d_kernel, d_output,
            inputRows, inputCols,
            kernelRows, kernelCols,
            outputRows, outputCols
        );
    }
    
    // 检查内核执行错误
    err = cudaGetLastError();
    CUDA_CHECK_ERROR(err);
    
    // 同步设备，确保计算完成
    err = cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(err);
    
    // 将结果从设备内存复制回主机内存
    err = cudaMemcpy(h_output, d_output, outputRows * outputCols * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(err);
    
    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

// 创建边缘检测卷积核
void create_edge_detection_kernel(float* kernel) {
    // 拉普拉斯算子卷积核
    float laplacian[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    
    for (int i = 0; i < 9; i++) {
        kernel[i] = laplacian[i];
    }
}

// 创建高斯平滑卷积核
void create_gaussian_kernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int center = size / 2;
    
    // 计算高斯函数值
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int x = i - center;
            int y = j - center;
            // 高斯函数：G(x,y) = (1/(2*pi*sigma^2)) * e^(-(x^2+y^2)/(2*sigma^2))
            float value = exp(-(x*x + y*y) / (2 * sigma * sigma)) / (2 * 3.14159f * sigma * sigma);
            kernel[i * size + j] = value;
            sum += value;
        }
    }
    
    // 归一化卷积核（确保所有元素和为1）
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

// 计算两个矩阵之间的最大差异
float matrix_diff(float* a, float* b, int rows, int cols) {
    float max_diff = 0.0f;
    
    for (int i = 0; i < rows * cols; i++) {
        float diff = fabs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    return max_diff;
}

// 计算执行时间 (毫秒)
double calculate_execution_time(clock_t start, clock_t end) {
    return ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
}

int main() {
    // 设置控制台代码页为简体中文GBK，以支持中文显示
    #ifdef _WIN32
    system("chcp 936 > nul");
    #endif
    
    printf("=== CUDA卷积计算程序 ===\n\n");
    
    // 设置矩阵尺寸
    int inputRows = 8, inputCols = 8;
    int kernelRows = 3, kernelCols = 3;
    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;
    
    // 输出CUDA设备信息
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CUDA_CHECK_ERROR(err);
    
    printf("检测到 %d 个CUDA设备\n", deviceCount);
    
    cudaDeviceProp deviceProp;
    for (int i = 0; i < deviceCount; i++) {
        err = cudaGetDeviceProperties(&deviceProp, i);
        CUDA_CHECK_ERROR(err);
        
        printf("设备 %d: %s\n", i, deviceProp.name);
        printf("  计算能力: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  多处理器数量: %d\n", deviceProp.multiProcessorCount);
        printf("  全局内存: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  最大线程数/块: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  最大线程数/多处理器: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  时钟频率: %.2f GHz\n\n", deviceProp.clockRate / 1000000.0f);
    }
    
    // 分配主机内存
    float* h_input = allocate_matrix_cpu(inputRows, inputCols);
    float* h_kernel = allocate_matrix_cpu(kernelRows, kernelCols);
    float* h_output_cuda = allocate_matrix_cpu(outputRows, outputCols);
    float* h_output_cpu = allocate_matrix_cpu(outputRows, outputCols);
    float* h_output_shared = allocate_matrix_cpu(outputRows, outputCols);
    
    // 生成随机输入矩阵
    create_random_matrix(h_input, inputRows, inputCols, 0.0f, 10.0f);
    
    // 创建边缘检测卷积核
    create_edge_detection_kernel(h_kernel);
    
    // 打印输入矩阵和卷积核
    print_matrix(h_input, inputRows, inputCols, "输入矩阵");
    print_matrix(h_kernel, kernelRows, kernelCols, "卷积核 (拉普拉斯算子)");
    
    // === CPU卷积计算 ===
    clock_t cpu_start = clock();
    convolve_cpu(h_input, h_kernel, h_output_cpu, inputRows, inputCols, kernelRows, kernelCols);
    clock_t cpu_end = clock();
    double cpu_time = calculate_execution_time(cpu_start, cpu_end);
    
    // === CUDA卷积计算 (基础版本) ===
    clock_t cuda_start = clock();
    convolve_cuda(h_input, h_kernel, h_output_cuda, inputRows, inputCols, kernelRows, kernelCols, false);
    clock_t cuda_end = clock();
    double cuda_time = calculate_execution_time(cuda_start, cuda_end);
    
    // === CUDA卷积计算 (共享内存优化版本) ===
    clock_t cuda_shared_start = clock();
    convolve_cuda(h_input, h_kernel, h_output_shared, inputRows, inputCols, kernelRows, kernelCols, true);
    clock_t cuda_shared_end = clock();
    double cuda_shared_time = calculate_execution_time(cuda_shared_start, cuda_shared_end);
    
    // 打印结果
    print_matrix(h_output_cpu, outputRows, outputCols, "CPU卷积结果");
    print_matrix(h_output_cuda, outputRows, outputCols, "CUDA卷积结果 (基础版本)");
    print_matrix(h_output_shared, outputRows, outputCols, "CUDA卷积结果 (共享内存版本)");
    
    // 验证结果
    float diff_cpu_cuda = matrix_diff(h_output_cpu, h_output_cuda, outputRows, outputCols);
    float diff_cpu_shared = matrix_diff(h_output_cpu, h_output_shared, outputRows, outputCols);
    
    printf("CPU与CUDA基础版本的最大差异: %f\n", diff_cpu_cuda);
    printf("CPU与CUDA共享内存版本的最大差异: %f\n", diff_cpu_shared);
    
    // 输出性能比较
    printf("\n=== 性能比较 ===\n");
    printf("CPU执行时间: %.2f 毫秒\n", cpu_time);
    printf("CUDA基础版本执行时间: %.2f 毫秒 (加速比: %.2fx)\n", 
           cuda_time, cpu_time / cuda_time);
    printf("CUDA共享内存版本执行时间: %.2f 毫秒 (加速比: %.2fx)\n", 
           cuda_shared_time, cpu_time / cuda_shared_time);
    
    // 尝试测试更大规模的矩阵
    printf("\n=== 大规模矩阵测试 ===\n");
    
    // 释放之前的内存
    free(h_input);
    free(h_kernel);
    free(h_output_cuda);
    free(h_output_cpu);
    free(h_output_shared);
    
    // 更大的测试矩阵
    inputRows = 1024;
    inputCols = 1024;
    kernelRows = 5;
    kernelCols = 5;
    outputRows = inputRows - kernelRows + 1;
    outputCols = inputCols - kernelCols + 1;
    
    // 重新分配内存
    h_input = allocate_matrix_cpu(inputRows, inputCols);
    h_kernel = allocate_matrix_cpu(kernelRows, kernelCols);
    h_output_cuda = allocate_matrix_cpu(outputRows, outputCols);
    h_output_cpu = allocate_matrix_cpu(outputRows, outputCols);
    h_output_shared = allocate_matrix_cpu(outputRows, outputCols);
    
    // 生成随机大矩阵和高斯卷积核
    create_random_matrix(h_input, inputRows, inputCols, 0.0f, 1.0f);
    create_gaussian_kernel(h_kernel, kernelRows, 1.0f);
    
    printf("大规模测试: 输入矩阵 %dx%d, 卷积核 %dx%d\n", inputRows, inputCols, kernelRows, kernelCols);
    
    // === CPU卷积计算 (大规模) ===
    cpu_start = clock();
    convolve_cpu(h_input, h_kernel, h_output_cpu, inputRows, inputCols, kernelRows, kernelCols);
    cpu_end = clock();
    cpu_time = calculate_execution_time(cpu_start, cpu_end);
    
    // === CUDA卷积计算 (基础版本, 大规模) ===
    cuda_start = clock();
    convolve_cuda(h_input, h_kernel, h_output_cuda, inputRows, inputCols, kernelRows, kernelCols, false);
    cuda_end = clock();
    cuda_time = calculate_execution_time(cuda_start, cuda_end);
    
    // === CUDA卷积计算 (共享内存优化版本, 大规模) ===
    cuda_shared_start = clock();
    convolve_cuda(h_input, h_kernel, h_output_shared, inputRows, inputCols, kernelRows, kernelCols, true);
    cuda_shared_end = clock();
    cuda_shared_time = calculate_execution_time(cuda_shared_start, cuda_shared_end);
    
    // 验证结果
    diff_cpu_cuda = matrix_diff(h_output_cpu, h_output_cuda, outputRows, outputCols);
    diff_cpu_shared = matrix_diff(h_output_cpu, h_output_shared, outputRows, outputCols);
    
    printf("大规模测试 - CPU与CUDA基础版本的最大差异: %f\n", diff_cpu_cuda);
    printf("大规模测试 - CPU与CUDA共享内存版本的最大差异: %f\n", diff_cpu_shared);
    
    // 输出大规模测试性能比较
    printf("\n=== 大规模测试性能比较 ===\n");
    printf("CPU执行时间: %.2f 毫秒\n", cpu_time);
    printf("CUDA基础版本执行时间: %.2f 毫秒 (加速比: %.2fx)\n", 
           cuda_time, cpu_time / cuda_time);
    printf("CUDA共享内存版本执行时间: %.2f 毫秒 (加速比: %.2fx)\n", 
           cuda_shared_time, cpu_time / cuda_shared_time);
    
    // 释放内存
    free(h_input);
    free(h_kernel);
    free(h_output_cuda);
    free(h_output_cpu);
    free(h_output_shared);
    
    // 重置CUDA设备
    cudaDeviceReset();
    
    printf("\n程序执行完毕\n");
    
    return 0;
} 