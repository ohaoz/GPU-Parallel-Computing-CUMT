// 文件编码：UTF-8 (无BOM)，在Windows上编译时使用GBK输出
/**
 * 文件名: convolution_c.c
 * 描述: 使用纯C语言实现的卷积计算
 * 
 * 卷积原理:
 * 卷积是信号处理和图像处理中的基本操作。在离散情况下，它表示为：
 * (f * g)[n] = Σ f[m] * g[n-m]
 * 
 * 在图像处理中，二维卷积公式为:
 * O[i,j] = Σ Σ I[i+m,j+n] * K[m,n]
 * 
 * 这里我们实现的是标准的二维卷积操作，用于图像处理。
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

/**
 * 为矩阵分配内存
 * @param rows 行数
 * @param cols 列数
 * @return 分配的二维浮点数矩阵
 */
float** allocate_matrix(int rows, int cols) {
    float** matrix = (float**)malloc(rows * sizeof(float*));
    if (!matrix) {
        fprintf(stderr, "Memory allocation failed\n"); // 内存分配失败
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
        if (!matrix[i]) {
            fprintf(stderr, "Memory allocation failed\n"); // 内存分配失败
            exit(EXIT_FAILURE);
        }
        // 初始化为0
        memset(matrix[i], 0, cols * sizeof(float));
    }
    
    return matrix;
}

/**
 * 释放矩阵内存
 * @param matrix 要释放的矩阵
 * @param rows 行数
 */
void free_matrix(float** matrix, int rows) {
    if (matrix) {
        for (int i = 0; i < rows; i++) {
            if (matrix[i]) {
                free(matrix[i]);
            }
        }
        free(matrix);
    }
}

/**
 * 创建随机矩阵
 * @param rows 行数
 * @param cols 列数
 * @param min_val 最小值
 * @param max_val 最大值
 * @return 生成的随机矩阵
 */
float** create_random_matrix(int rows, int cols, float min_val, float max_val) {
    float** matrix = allocate_matrix(rows, cols);

    // 初始化随机数生成器(只初始化一次，避免短时间内重复调用导致相同序列)
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 生成范围在[min_val, max_val]的随机浮点数
            float random_val = min_val + ((float)rand() / RAND_MAX) * (max_val - min_val);
            matrix[i][j] = random_val;
        }
    }
    
    return matrix;
}

/**
 * 打印矩阵
 * @param matrix 要打印的矩阵
 * @param rows 行数
 * @param cols 列数
 * @param name 矩阵名称
 */
void print_matrix(float** matrix, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * 执行卷积计算
 * @param input 输入矩阵
 * @param input_rows 输入矩阵行数
 * @param input_cols 输入矩阵列数
 * @param kernel 卷积核
 * @param kernel_rows 卷积核行数
 * @param kernel_cols 卷积核列数
 * @return 卷积结果矩阵
 */
float** convolve(float** input, int input_rows, int input_cols, 
                float** kernel, int kernel_rows, int kernel_cols) {
    // 计算输出矩阵的尺寸
    // 这里使用"有效"卷积，即只在卷积核完全在输入图像内部时计算
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    
    if (output_rows <= 0 || output_cols <= 0) {
        fprintf(stderr, "Convolution kernel size is larger than input matrix, cannot compute convolution\n"); // 卷积核尺寸大于输入矩阵，无法计算卷积
        return NULL;
    }
    
    // 分配输出矩阵内存
    float** output = allocate_matrix(output_rows, output_cols);
    
    // 计算卷积
    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            // 对当前位置应用卷积核
            float sum = 0.0f;
            for (int ki = 0; ki < kernel_rows; ki++) {
                for (int kj = 0; kj < kernel_cols; kj++) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
    
    return output;
}

/**
 * 边缘检测卷积示例
 * @param input 输入矩阵
 * @param input_rows 输入矩阵行数
 * @param input_cols 输入矩阵列数
 * @return 处理后的矩阵
 */
float** detect_edges(float** input, int input_rows, int input_cols) {
    // 创建拉普拉斯算子卷积核用于边缘检测
    float laplacian[3][3] = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };
    
    // 将拉普拉斯算子转换为动态分配的矩阵
    float** kernel = allocate_matrix(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            kernel[i][j] = laplacian[i][j];
        }
    }
    
    // 执行卷积
    float** result = convolve(input, input_rows, input_cols, kernel, 3, 3);
    
    // 释放卷积核内存
    free_matrix(kernel, 3);
    
    return result;
}

/**
 * 创建高斯平滑卷积核
 * @param size 卷积核大小（必须是奇数）
 * @param sigma 高斯分布的标准差
 * @return 高斯卷积核
 */
float** create_gaussian_kernel(int size, float sigma) {
    if (size % 2 == 0) {
        fprintf(stderr, "Kernel size must be odd\n"); // 卷积核大小必须是奇数
        return NULL;
    }
    
    float** kernel = allocate_matrix(size, size);
    float sum = 0.0f;
    int center = size / 2;
    
    // 计算高斯函数值
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int x = i - center;
            int y = j - center;
            // 高斯函数：G(x,y) = (1/(2*pi*sigma^2)) * e^(-(x^2+y^2)/(2*sigma^2))
            kernel[i][j] = exp(-(x*x + y*y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            sum += kernel[i][j];
        }
    }
    
    // 归一化卷积核（确保所有元素和为1）
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[i][j] /= sum;
        }
    }
    
    return kernel;
}

/**
 * 计算两个矩阵之间的最大差异
 * @param a 第一个矩阵
 * @param b 第二个矩阵
 * @param rows 行数
 * @param cols 列数
 * @return 最大差异值
 */
float matrix_diff(float** a, float** b, int rows, int cols) {
    float max_diff = 0.0f;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float diff = fabs(a[i][j] - b[i][j]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    
    return max_diff;
}

/**
 * 主函数 - 演示卷积的使用
 */
int main() {
    // 设置控制台代码页为简体中文GBK，以支持中文显示
    #ifdef _WIN32
    system("chcp 936 > nul");
    #endif
    
    // 设置输入矩阵和卷积核大小
    const int INPUT_ROWS = 10;
    const int INPUT_COLS = 10;
    const int KERNEL_ROWS = 3;
    const int KERNEL_COLS = 3;
    
    // 创建随机输入矩阵
    float** input = create_random_matrix(INPUT_ROWS, INPUT_COLS, 0.0f, 10.0f);
    
    // 创建卷积核 - 边缘检测
    float** edge_kernel = allocate_matrix(KERNEL_ROWS, KERNEL_COLS);
    // 初始化拉普拉斯算子卷积核
    float laplacian[3][3] = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };
    for (int i = 0; i < KERNEL_ROWS; i++) {
        for (int j = 0; j < KERNEL_COLS; j++) {
            edge_kernel[i][j] = laplacian[i][j];
        }
    }
    
    // 打印输入矩阵和卷积核
    print_matrix(input, INPUT_ROWS, INPUT_COLS, "Input Matrix"); // 输入矩阵
    print_matrix(edge_kernel, KERNEL_ROWS, KERNEL_COLS, "Edge Detection Kernel"); // 边缘检测卷积核
    
    // 测量卷积计算时间
    clock_t start = clock();
    float** result = convolve(input, INPUT_ROWS, INPUT_COLS, edge_kernel, KERNEL_ROWS, KERNEL_COLS);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC * 1000.0; // 毫秒
    
    // 计算输出矩阵的尺寸
    int output_rows = INPUT_ROWS - KERNEL_ROWS + 1;
    int output_cols = INPUT_COLS - KERNEL_COLS + 1;
    
    // 打印结果和性能信息
    print_matrix(result, output_rows, output_cols, "Convolution Result"); // 卷积结果
    printf("Convolution computation time: %.2f milliseconds\n\n", time_spent); // 卷积计算时间(毫秒)
    
    // 创建高斯模糊卷积核示例
    printf("Gaussian Blur Example:\n"); // 高斯模糊示例
    float** gaussian_kernel = create_gaussian_kernel(5, 1.0f);
    print_matrix(gaussian_kernel, 5, 5, "Gaussian Kernel (sigma=1.0)"); // 高斯卷积核
    
    // 应用高斯模糊
    float** blurred = convolve(input, INPUT_ROWS, INPUT_COLS, gaussian_kernel, 5, 5);
    print_matrix(blurred, INPUT_ROWS - 5 + 1, INPUT_COLS - 5 + 1, "Gaussian Blur Result"); // 高斯模糊结果
    
    // 边缘检测示例
    printf("Edge Detection Example:\n"); // 边缘检测示例
    float** edges = detect_edges(input, INPUT_ROWS, INPUT_COLS);
    print_matrix(edges, INPUT_ROWS - 3 + 1, INPUT_COLS - 3 + 1, "Detected Edges"); // 检测到的边缘
    
    // 释放所有分配的内存
    free_matrix(input, INPUT_ROWS);
    free_matrix(edge_kernel, KERNEL_ROWS);
    free_matrix(result, output_rows);
    free_matrix(gaussian_kernel, 5);
    free_matrix(blurred, INPUT_ROWS - 5 + 1);
    free_matrix(edges, INPUT_ROWS - 3 + 1);
    
    return 0;
} 