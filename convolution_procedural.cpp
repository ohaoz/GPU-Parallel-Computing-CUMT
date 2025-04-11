// 文件编码：UTF-8 (无BOM)，在Windows上编译时使用GBK输出
/**
 * 文件名: convolution_procedural.cpp
 * 描述: 使用面向过程的方法实现2D卷积计算
 * 
 * 卷积原理简介:
 * 卷积是图像处理和信号处理中的基本操作，用于提取特征、平滑、锐化等。
 * 在2D图像处理中，卷积涉及将一个小的矩阵（卷积核或滤波器）滑过图像，
 * 对每个位置的重叠区域执行元素间乘积再求和的操作。
 * 
 * 数学表达式: 对于输入图像I和卷积核K，输出O的计算公式为:
 * O[i,j] = ∑∑ I[i+m,j+n] * K[m,n]
 * 
 * 面向过程实现思路:
 * 1. 表示矩阵的简单数据结构
 * 2. 读取输入数据和卷积核
 * 3. 进行卷积计算
 * 4. 输出结果
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

/**
 * 简单矩阵结构体 - 存储二维浮点数矩阵
 */
struct Matrix {
    int rows;                               // 行数
    int cols;                               // 列数
    std::vector<std::vector<float>> data;   // 矩阵数据
    
    // 构造简单矩阵
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(r, std::vector<float>(c, 0.0f));
    }
};

/**
 * 随机填充矩阵
 * @param mat 要填充的矩阵
 * @param min_val 随机值下限
 * @param max_val 随机值上限
 */
void fill_random(Matrix& mat, float min_val = 0.0f, float max_val = 1.0f) {
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            float random_val = min_val + static_cast<float>(rand()) / RAND_MAX * (max_val - min_val);
            mat.data[i][j] = random_val;
        }
    }
}

/**
 * 创建特定的卷积核
 * @param kernel_type 卷积核类型: "edge" - 边缘检测, "blur" - 模糊, "sharpen" - 锐化
 * @return 创建的卷积核矩阵
 */
Matrix create_kernel(const std::string& kernel_type) {
    if (kernel_type == "edge") {
        // 拉普拉斯算子 - 边缘检测
        Matrix kernel(3, 3);
        float edge_data[3][3] = {
            {-1, -1, -1},
            {-1,  8, -1},
            {-1, -1, -1}
        };
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                kernel.data[i][j] = edge_data[i][j];
            }
        }
        return kernel;
    } 
    else if (kernel_type == "blur") {
        // 简单的均值滤波 - 模糊
        Matrix kernel(3, 3);
        float blur_value = 1.0f / 9.0f;  // 所有元素和为1，确保不改变整体亮度
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                kernel.data[i][j] = blur_value;
            }
        }
        return kernel;
    } 
    else if (kernel_type == "sharpen") {
        // 锐化卷积核
        Matrix kernel(3, 3);
        float sharpen_data[3][3] = {
            { 0, -1,  0},
            {-1,  5, -1},
            { 0, -1,  0}
        };
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                kernel.data[i][j] = sharpen_data[i][j];
            }
        }
        return kernel;
    } 
    else {
        // 默认返回单位卷积核（不改变输入）
        Matrix kernel(3, 3);
        kernel.data[1][1] = 1.0f; // 中心元素设为1，其他保持0
        return kernel;
    }
}

/**
 * 打印矩阵内容
 * @param mat 要打印的矩阵
 * @param name 矩阵名称
 */
void print_matrix(const Matrix& mat, const std::string& name) {
    std::cout << name << " (" << mat.rows << "x" << mat.cols << "):" << std::endl;
    
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(6) << mat.data[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * 执行2D卷积计算
 * @param input 输入矩阵
 * @param kernel 卷积核
 * @return 卷积结果矩阵
 */
Matrix convolve(const Matrix& input, const Matrix& kernel) {
    // 计算输出矩阵的尺寸 (有效卷积模式)
    int output_rows = input.rows - kernel.rows + 1;
    int output_cols = input.cols - kernel.cols + 1;
    
    // 创建输出矩阵
    Matrix output(output_rows, output_cols);
    
    // 进行卷积计算
    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            float sum = 0.0f;
            
            // 应用卷积核
            for (int ki = 0; ki < kernel.rows; ++ki) {
                for (int kj = 0; kj < kernel.cols; ++kj) {
                    sum += input.data[i + ki][j + kj] * kernel.data[ki][kj];
                }
            }
            
            output.data[i][j] = sum;
        }
    }
    
    return output;
}

/**
 * 创建高斯卷积核
 * @param size 卷积核大小 (应为奇数)
 * @param sigma 高斯分布的标准差
 * @return 高斯卷积核
 */
Matrix create_gaussian_kernel(int size, float sigma) {
    Matrix kernel(size, size);
    float sum = 0.0f;
    int center = size / 2;
    
    // 计算高斯卷积核
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            // 计算当前位置到中心的距离平方
            float x = float(i - center);
            float y = float(j - center);
            float distance_squared = x * x + y * y;
            
            // 高斯函数
            kernel.data[i][j] = std::exp(-distance_squared / (2.0f * sigma * sigma));
            sum += kernel.data[i][j];
        }
    }
    
    // 归一化，使所有元素和为1
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel.data[i][j] /= sum;
        }
    }
    
    return kernel;
}

/**
 * 对图像矩阵执行特定滤镜
 * @param input 输入图像矩阵
 * @param filter_type 滤镜类型: "edge", "blur", "sharpen", "gaussian"
 * @return 处理后的图像矩阵
 */
Matrix apply_filter(const Matrix& input, const std::string& filter_type) {
    Matrix kernel(3, 3);
    
    if (filter_type == "gaussian") {
        kernel = create_gaussian_kernel(5, 1.0f);
    } else {
        kernel = create_kernel(filter_type);
    }
    
    std::cout << "Applying " << filter_type << " filter:" << std::endl; // 应用滤镜
    print_matrix(kernel, "Filter Kernel"); // 使用的卷积核
    
    return convolve(input, kernel);
}

/**
 * 计算卷积结果的绝对值(用于边缘检测后显示)
 * @param mat 输入矩阵
 * @return 绝对值矩阵
 */
Matrix abs_matrix(const Matrix& mat) {
    Matrix result(mat.rows, mat.cols);
    
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            result.data[i][j] = std::fabs(mat.data[i][j]);
        }
    }
    
    return result;
}

/**
 * 归一化矩阵值到指定范围
 * @param mat 要归一化的矩阵
 * @param min_val 目标最小值
 * @param max_val 目标最大值
 */
void normalize_matrix(Matrix& mat, float min_val = 0.0f, float max_val = 1.0f) {
    // 查找当前的最大和最小值
    float current_min = mat.data[0][0];
    float current_max = mat.data[0][0];
    
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            if (mat.data[i][j] < current_min) current_min = mat.data[i][j];
            if (mat.data[i][j] > current_max) current_max = mat.data[i][j];
        }
    }
    
    // 避免除以零
    if (current_max == current_min) {
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                mat.data[i][j] = min_val;
            }
        }
        return;
    }
    
    // 归一化
    float scale = (max_val - min_val) / (current_max - current_min);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            mat.data[i][j] = min_val + (mat.data[i][j] - current_min) * scale;
        }
    }
}

/**
 * 主函数 - 演示面向过程的卷积实现
 */
int main() {
    // 设置控制台代码页为简体中文GBK，以支持中文显示
    #ifdef _WIN32
    system("chcp 936 > nul");
    #endif
    
    // 初始化随机数生成器
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    // 创建测试输入矩阵
    const int INPUT_SIZE = 8;
    Matrix input(INPUT_SIZE, INPUT_SIZE);
    fill_random(input, 0.0f, 10.0f);
    
    // 打印输入矩阵
    print_matrix(input, "Input Matrix"); // 输入矩阵
    
    // 1. 边缘检测示例
    Matrix edges = apply_filter(input, "edge");
    print_matrix(edges, "Edge Detection Result"); // 边缘检测结果
    
    // 取边缘检测结果的绝对值(通常边缘检测需要这样处理以便更好地可视化)
    Matrix abs_edges = abs_matrix(edges);
    normalize_matrix(abs_edges, 0.0f, 10.0f);  // 归一化到与原图相同的范围
    print_matrix(abs_edges, "Edge Detection Result (Absolute Values)"); // 边缘检测结果(绝对值)
    
    // 2. 模糊效果示例
    Matrix blurred = apply_filter(input, "blur");
    print_matrix(blurred, "Blur Effect Result"); // 模糊处理结果
    
    // 3. 锐化效果示例
    Matrix sharpened = apply_filter(input, "sharpen");
    print_matrix(sharpened, "Sharpening Effect Result"); // 锐化处理结果
    
    // 4. 高斯模糊示例
    Matrix gaussian_blurred = apply_filter(input, "gaussian");
    print_matrix(gaussian_blurred, "Gaussian Blur Result"); // 高斯模糊结果
    
    // 5. 自定义卷积核示例
    std::cout << "Custom Kernel Example:" << std::endl; // 自定义卷积核示例
    Matrix custom_kernel(3, 3);
    // 创建一个水平方向Sobel算子(检测水平边缘)
    float sobel_h[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            custom_kernel.data[i][j] = sobel_h[i][j];
        }
    }
    
    print_matrix(custom_kernel, "Horizontal Sobel Operator"); // 水平Sobel算子
    Matrix sobel_result = convolve(input, custom_kernel);
    Matrix abs_sobel = abs_matrix(sobel_result);
    normalize_matrix(abs_sobel, 0.0f, 10.0f);
    print_matrix(abs_sobel, "Horizontal Edge Detection Result"); // 水平边缘检测结果
    
    return 0;
} 