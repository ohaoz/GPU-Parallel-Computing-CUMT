// 文件编码：UTF-8 (无BOM)，在Windows上编译时使用GBK输出
/**
 * 文件名: convolution_cpp_parallel.cpp
 * 描述: 使用C++11标准库中的多线程功能实现并行卷积计算
 * 
 * 卷积原理:
 * 卷积是信号处理和图像处理中的基本操作，数学上表示为两个函数的积的积分。
 * 在离散情况下，卷积计算公式为:
 * (f * g)[n] = Σ f[m] * g[n-m]
 * 
 * 在图像处理中，卷积通常用于应用滤波器(卷积核)到图像上，公式为:
 * O[i,j] = Σ Σ I[i+m,j+n] * K[m,n]
 * 
 * 并行化思路:
 * 1. 将输入数据划分为多个区域，分配给不同的线程
 * 2. 每个线程独立计算自己负责区域的卷积结果
 * 3. 所有线程完成后合并结果
 */

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <chrono>
#include <iomanip>
#include <algorithm>

// 定义卷积计算函数的类型别名，增强代码可读性
using ConvolutionFunction = std::function<void(
    const std::vector<std::vector<float>>&,  // 输入矩阵
    const std::vector<std::vector<float>>&,  // 卷积核
    std::vector<std::vector<float>>&,        // 输出矩阵
    int, int, int, int)>;                   // 起始行、结束行、起始列、结束列

// 互斥锁，用于线程安全地打印信息
std::mutex print_mutex;

/**
 * 打印矩阵函数
 * @param matrix 要打印的矩阵
 * @param name 矩阵名称
 */
void print_matrix(const std::vector<std::vector<float>>& matrix, const std::string& name) {
    std::lock_guard<std::mutex> lock(print_mutex);
    std::cout << name << " (" << matrix.size() << "x" << matrix[0].size() << "):" << std::endl;
    
    for (const auto& row : matrix) {
        for (float val : row) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(6) << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * 单线程卷积计算(用于对比)
 * @param input 输入矩阵
 * @param kernel 卷积核
 * @return 卷积结果矩阵
 */
std::vector<std::vector<float>> convolve_sequential(
    const std::vector<std::vector<float>>& input,
    const std::vector<std::vector<float>>& kernel) {
    
    // 获取矩阵和卷积核的尺寸
    int input_rows = input.size();
    int input_cols = input[0].size();
    int kernel_rows = kernel.size();
    int kernel_cols = kernel[0].size();
    
    // 计算输出矩阵的尺寸
    // 这里使用"有效"卷积，即只在卷积核完全在输入图像内部时计算
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    
    // 创建并初始化输出矩阵
    std::vector<std::vector<float>> output(output_rows, std::vector<float>(output_cols, 0.0f));
    
    // 计算卷积
    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            // 对当前位置应用卷积核
            for (int ki = 0; ki < kernel_rows; ++ki) {
                for (int kj = 0; kj < kernel_cols; ++kj) {
                    output[i][j] += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
        }
    }
    
    return output;
}

/**
 * 执行部分卷积计算的工作函数(供线程调用)
 * @param input 输入矩阵
 * @param kernel 卷积核
 * @param output 输出矩阵(引用，直接修改)
 * @param start_row 起始行
 * @param end_row 结束行
 * @param start_col 起始列
 * @param end_col 结束列
 */
void convolve_worker(
    const std::vector<std::vector<float>>& input,
    const std::vector<std::vector<float>>& kernel,
    std::vector<std::vector<float>>& output,
    int start_row, int end_row, int start_col, int end_col) {
    
    int kernel_rows = kernel.size();
    int kernel_cols = kernel[0].size();
    
    // 计算分配给此线程的区域的卷积
    for (int i = start_row; i < end_row; ++i) {
        for (int j = start_col; j < end_col; ++j) {
            output[i][j] = 0.0f;  // 确保初始值为0
            
            // 应用卷积核
            for (int ki = 0; ki < kernel_rows; ++ki) {
                for (int kj = 0; kj < kernel_cols; ++kj) {
                    output[i][j] += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
        }
    }
}

/**
 * 多线程并行卷积实现
 * @param input 输入矩阵
 * @param kernel 卷积核
 * @param num_threads 线程数量(默认使用系统支持的最大线程数)
 * @return 卷积结果矩阵
 */
std::vector<std::vector<float>> convolve_parallel(
    const std::vector<std::vector<float>>& input,
    const std::vector<std::vector<float>>& kernel,
    int num_threads = std::thread::hardware_concurrency()) {
    
    // 获取矩阵和卷积核的尺寸
    int input_rows = input.size();
    int input_cols = input[0].size();
    int kernel_rows = kernel.size();
    int kernel_cols = kernel[0].size();
    
    // 计算输出矩阵的尺寸
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    
    // 创建并初始化输出矩阵
    std::vector<std::vector<float>> output(output_rows, std::vector<float>(output_cols, 0.0f));
    
    // 调整线程数量，确保不超过工作量
    num_threads = std::min(num_threads, output_rows * output_cols);
    if (num_threads <= 0) num_threads = 1;
    
    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "Using " << num_threads << " threads for parallel convolution computation" << std::endl;
    }
    
    std::vector<std::thread> threads;
    
    // 确定如何划分工作负载
    // 这里使用简单的行划分策略，可以根据实际情况优化
    int rows_per_thread = output_rows / num_threads;
    int remaining_rows = output_rows % num_threads;
    
    int start_row = 0;
    
    // 创建并启动工作线程
    for (int t = 0; t < num_threads; ++t) {
        // 计算当前线程负责的行数
        int thread_rows = rows_per_thread + (t < remaining_rows ? 1 : 0);
        int end_row = start_row + thread_rows;
        
        // 创建线程并分配工作
        threads.emplace_back(convolve_worker, 
                            std::ref(input), std::ref(kernel), std::ref(output),
                            start_row, end_row, 0, output_cols);
        
        start_row = end_row;
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    return output;
}

/**
 * 使用卷积核对输入图像进行边缘检测的示例
 * @param input 输入图像矩阵
 * @return 处理后的图像矩阵
 */
std::vector<std::vector<float>> detect_edges(const std::vector<std::vector<float>>& input) {
    // Sobel算子 - 用于边缘检测的常用卷积核
    std::vector<std::vector<float>> sobel_x = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    // 使用并行卷积实现边缘检测
    return convolve_parallel(input, sobel_x);
}

/**
 * 生成测试用的随机矩阵
 * @param rows 行数
 * @param cols 列数
 * @param min_val 最小值
 * @param max_val 最大值
 * @return 生成的随机矩阵
 */
std::vector<std::vector<float>> generate_random_matrix(int rows, int cols, float min_val = 0.0f, float max_val = 1.0f) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));

    // 设置随机数生成器(只初始化一次，避免快速重复调用产生相同序列)
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned int>(time(nullptr)));
        seeded = true;
    }
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float random_val = min_val + static_cast<float>(rand()) / RAND_MAX * (max_val - min_val);
            matrix[i][j] = random_val;
        }
    }
    
    return matrix;
}

/**
 * 计算矩阵差异，用于验证结果
 * @param a 第一个矩阵
 * @param b 第二个矩阵
 * @return 差异的最大绝对值
 */
float matrix_difference(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        return std::numeric_limits<float>::max();
    }
    
    float max_diff = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            max_diff = std::max(max_diff, std::abs(a[i][j] - b[i][j]));
        }
    }
    
    return max_diff;
}

/**
 * 主函数 - 演示并行卷积的使用
 */
int main() {
    // 设置控制台代码页为简体中文GBK，以支持中文显示
    #ifdef _WIN32
    system("chcp 936 > nul");
    #endif
    
    // 设置输入矩阵大小和卷积核大小
    const int INPUT_ROWS = 10;
    const int INPUT_COLS = 10;
    const int KERNEL_ROWS = 3;
    const int KERNEL_COLS = 3;
    
    // 生成随机输入矩阵
    auto input = generate_random_matrix(INPUT_ROWS, INPUT_COLS, 0.0f, 10.0f);
    
    // 创建卷积核 - 使用边缘检测卷积核作为示例
    std::vector<std::vector<float>> kernel = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };
    
    // 打印输入矩阵和卷积核
    print_matrix(input, "Input Matrix"); // 输入矩阵
    print_matrix(kernel, "Convolution Kernel"); // 卷积核
    
    // 测量单线程版本性能
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result_sequential = convolve_sequential(input, kernel);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_sequential = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // 测量多线程版本性能
    start_time = std::chrono::high_resolution_clock::now();
    auto result_parallel = convolve_parallel(input, kernel);
    end_time = std::chrono::high_resolution_clock::now();
    auto duration_parallel = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // 打印结果
    print_matrix(result_sequential, "Sequential Convolution Result"); // 单线程卷积结果
    print_matrix(result_parallel, "Parallel Convolution Result"); // 多线程卷积结果
    
    // 验证两种实现的结果是否一致
    float diff = matrix_difference(result_sequential, result_parallel);
    std::cout << "Result difference: " << diff << std::endl; // 结果差异
    
    // 打印性能对比
    std::cout << "Sequential computation time: " << duration_sequential.count() << " microseconds" << std::endl; // 单线程计算时间(微秒)
    std::cout << "Parallel computation time: " << duration_parallel.count() << " microseconds" << std::endl; // 多线程计算时间(微秒)
    
    if (duration_sequential.count() > 0) {
        float speedup = static_cast<float>(duration_sequential.count()) / duration_parallel.count();
        std::cout << "Speedup: " << speedup << "x" << std::endl; // 加速比
    }
    
    // 示例：边缘检测应用
    std::cout << "\nEdge Detection Example:" << std::endl; // 边缘检测示例
    auto edges = detect_edges(input);
    print_matrix(edges, "Detected Edges"); // 检测到的边缘
    
    return 0;
} 