# GPU并行计算 - CUDA卷积实现

## 项目概述

本项目是计算机学院"GPU并行计算"课程的实验作业，主要展示了使用CUDA实现的卷积计算，并与CPU实现进行性能对比。项目包含四种不同的卷积实现：

1. **C语言串行版本** (`convolution_c.c`)
2. **C++面向过程版本** (`convolution_procedural.cpp`)
3. **C++多线程并行版本** (`convolution_cpp_parallel.cpp`)
4. **CUDA GPU加速版本** (`convolution_cuda.cu`)

通过对比这四种实现，可以直观地了解CPU串行计算、CPU多线程并行计算与GPU大规模并行计算之间的性能差异。

## 项目结构

```
/
├── convolution_c.c                  # C语言卷积实现
├── convolution_procedural.cpp       # C++面向过程卷积实现
├── convolution_cpp_parallel.cpp     # C++多线程并行卷积实现
├── convolution_cuda.cu              # CUDA GPU加速卷积实现
├── compile_cuda.bat                 # CUDA编译和运行脚本
├── fix_encoding.bat                 # 解决中文编码问题的脚本
├── run_convolution.bat              # 运行C/C++实现的脚本
├── CUDA_README.md                   # CUDA实现的详细说明
└── README_CUDA.md                   # 本文档，提供项目整体说明
```

## 卷积计算原理

卷积是信号处理和图像处理中的基本操作，在图像处理中用于实现模糊、锐化、边缘检测等效果。二维卷积的数学表达式为：

$$O[i,j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I[i+m,j+n] \times K[m,n]$$

其中：
- I 是输入图像/矩阵
- K 是卷积核/滤波器
- O 是输出图像/矩阵

## 四种实现的比较

| 特性 | C语言版本 | C++面向过程版本 | C++多线程版本 | CUDA GPU版本 |
|------|---------|--------------|------------|------------|
| 编程范式 | 过程式 | 面向过程的C++ | CPU多线程并行 | GPU大规模并行 |
| 内存管理 | 手动(malloc/free) | 自动(C++容器) | 自动(C++容器) | CUDA内存模型 |
| 并行能力 | 单线程 | 单线程 | 多线程 | 数千核心并行 |
| 性能优势 | 内存效率最高 | 代码简洁易读 | 多核CPU加速 | 大规模矩阵最快 |
| 适用场景 | 资源受限环境 | 学习和原型开发 | 多核CPU系统 | 需要极高性能 |
| 代码复杂度 | 中等 | 较低 | 较高 | 高 |
| 优化难度 | 低 | 低 | 中 | 高 |
| 可扩展性 | 一般 | 好 | 很好 | 极好 |

## CUDA实现的特点

CUDA版本的卷积实现具有以下特点：

1. **大规模并行**：每个CUDA线程负责计算输出矩阵中的一个元素，能够同时执行数千个线程。

2. **内存优化**：提供了基础版本和共享内存优化版本两种实现。共享内存版本通过将卷积核加载到共享内存中减少全局内存访问，提高计算效率。

3. **性能对比**：同时实现了CPU版本，可以直观比较CPU和GPU的性能差异。在大规模矩阵(如1024x1024)上，GPU版本通常能达到CPU版本数十倍甚至上百倍的性能提升。

4. **自动性能测试**：程序会自动计算并输出不同实现的执行时间和加速比，便于性能分析。

## 如何运行CUDA版本

### 前提条件

1. 安装NVIDIA CUDA Toolkit
2. 确保系统中有兼容的NVIDIA GPU
3. 将NVCC编译器添加到系统环境变量中

### 编译和运行

直接运行`compile_cuda.bat`批处理文件：

```
compile_cuda.bat
```

### 手动编译

也可以手动执行以下命令进行编译：

```
nvcc -o convolution_cuda convolution_cuda.cu -Xcompiler "/EHsc /W3" -arch=sm_30 -fexec-charset=GBK -finput-charset=UTF-8
```

然后运行编译好的程序：

```
convolution_cuda.exe
```

## 如何运行C/C++版本

运行`fix_encoding.bat`或`run_convolution.bat`批处理文件：

```
fix_encoding.bat
```

或手动编译：

```
chcp 936
g++ -o convolution_parallel -fexec-charset=GBK -finput-charset=UTF-8 convolution_cpp_parallel.cpp -std=c++11 -pthread
gcc -o convolution_c -fexec-charset=GBK -finput-charset=UTF-8 convolution_c.c -lm
g++ -o convolution_procedural -fexec-charset=GBK -finput-charset=UTF-8 convolution_procedural.cpp -std=c++11
```

## 性能测试结果

> 注意：以下数据为示例数据，实际性能取决于硬件配置

### 小规模矩阵 (8x8, 卷积核3x3)

| 实现方式 | 执行时间 | 相对CPU加速比 |
|---------|--------|------------|
| CPU串行版本 | 0.15 ms | 1.0x |
| C++多线程(8线程) | 0.08 ms | 1.9x |
| CUDA基础版本 | 0.05 ms | 3.0x |
| CUDA共享内存版本 | 0.04 ms | 3.8x |

### 大规模矩阵 (1024x1024, 卷积核5x5)

| 实现方式 | 执行时间 | 相对CPU加速比 |
|---------|--------|------------|
| CPU串行版本 | 850.32 ms | 1.0x |
| C++多线程(8线程) | 125.64 ms | 6.8x |
| CUDA基础版本 | 10.25 ms | 83.0x |
| CUDA共享内存版本 | 7.53 ms | 112.9x |

## 优化方向

CUDA实现可以进一步优化的方向包括：

1. **更高级的内存访问模式**：使用纹理内存或常量内存存储卷积核
2. **输入数据的共享内存优化**：除卷积核外，也可将输入数据块加载到共享内存中
3. **循环展开**：对内层循环进行展开以减少循环开销
4. **合理选择线程块大小**：根据不同的GPU架构调整最优的线程块大小
5. **使用统一内存**：简化内存管理，但可能影响性能
6. **流水线执行**：使用CUDA Streams实现计算与内存操作的重叠

## 中文显示与编码问题

为避免中文显示乱码，本项目提供了以下解决方案：

1. **GBK编码输出**：使用编译选项`-fexec-charset=GBK -finput-charset=UTF-8`确保中文正常显示
2. **批处理文件设置**：在批处理文件中使用`chcp 936`设置控制台为简体中文编码

## 注意事项

1. CUDA程序需要NVIDIA GPU和CUDA Toolkit支持
2. 编译参数`-arch=sm_30`表示支持计算能力3.0及以上的GPU，可根据实际设备调整
3. 大规模矩阵测试可能需要较大显存，如遇问题可调小矩阵尺寸

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Optimizing CUDA Programs](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#optimizing-cuda-applications)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples) 