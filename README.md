# GPU-Parallel-Computing-CUMT

## 项目简介

> 实验一和实验二都是要求计算卷积

### 实验一
了解掌握多核并行程序设计，理解**CPU多核并发编程模式**，进程与线程的概念，线程间通信（同步与互斥），包括：互斥函数、临界区（临界段），生产者与消费者同步等算法

### 实验二
掌握*CUDA C / CUDA C++*，基于多核**CPU**和**GPU**的**异构CUDA编程**模型，CUDA的编程优化，掌握优化CUDA并行计算的应用程序设计

## Deadline
2025-04-15 第八周

# 在 Windows 上安装 MinGW-w64 (g++ 编译器) 教程

我们推荐使用 MSYS2 来安装和管理 MinGW-w64，因为它提供了一个方便的包管理器。

## 步骤概览

1. 下载并安装 MSYS2
2. 更新 MSYS2 基础包
3. 通过 MSYS2 安装 MinGW-w64 GCC 工具链 (包含 g++)
4. 将 MinGW-w64 的 bin 目录添加到 Windows 的 PATH 环境变量
5. 验证安装

## 详细步骤

### 1. 下载并安装 MSYS2

- 访问 [MSYS2 官方网站](https://www.msys2.org/)
- 在首页找到 "Installation" 部分，下载最新的 .exe 安装程序
- 运行下载的安装程序，按照提示进行安装
- 建议保持默认的安装路径 (例如 C:\msys64)
- 安装完成后，勾选 "Run MSYS2 now" 并点击 "Finish"

### 2. 更新 MSYS2 基础包

- MSYS2 安装完成后会打开一个类似 Linux 终端的窗口 (MSYS2 MSYS)
- 在这个窗口中，首先更新包数据库和核心系统包
 ```
  pacman -Syu
 ```
- 它可能会提示你关闭窗口，按 Y 然后按 Enter，手动关闭 MSYS2 窗口
- 从 Windows 开始菜单重新启动 MSYS2 (找到 "MSYS2 MSYS" 并运行)
- 再次运行更新命令，以确保所有基础包都是最新的
 ```
  pacman -Su
 ```
- 根据提示完成更新

### 3. 安装 MinGW-w64 GCC 工具链

- 在 MSYS2 MSYS 窗口中，安装适用于 64 位 Windows 的 MinGW-w64 GCC 工具链
- 系统会列出将要安装的包 (包括 gcc, g++, make 等)
```
pacman -S --needed base-devel mingw-w64-x86_64-toolchain
```
- 当询问选择时，直接按 Enter (选择全部)
- 当询问是否继续安装时，输入 Y 并按 Enter
- 等待安装过程完成

### 4. 添加 MinGW-w64 到 Windows PATH

- 找到 MinGW-w64 的 bin 目录：
  - 默认安装路径通常为 C:\msys64\mingw64\bin
- 添加到 PATH 环境变量：
  - 在 Windows 搜索栏中搜索 "环境变量"，打开系统属性
  - 点击 "高级" 选项卡下的 "环境变量" 按钮
  - 在 "用户变量" 中选中 Path，然后点击 "编辑"
  - 点击 "新建"，将 MinGW-w64 bin 目录路径粘贴进去
  - 点击 "确定" 关闭所有窗口

### 5. 验证安装

- 关闭所有已经打开的 PowerShell 或 CMD 窗口
- 打开一个新的 PowerShell 或 CMD 窗口
- 输入 `g++ --version` 命令
- 如果看到版本信息输出，说明 g++ 安装成功并且 PATH 配置正确

现在你的 Windows 系统已经正确安装并配置了 g++ 编译器，可以回到项目目录尝试编译命令了。

# 在Windows安装CUDA


CUDA（Compute Unified Device Architecture）是NVIDIA推出的并行计算平台和编程模型，它允许开发者利用NVIDIA GPU的强大计算能力来加速应用程序。以下是在Windows系统上安装CUDA的详细步骤。

## 前提条件

1. 确保你的电脑配备了NVIDIA GPU
2. 安装了兼容的Windows操作系统（Windows 10或Windows 11）
3. 安装了最新的NVIDIA显卡驱动

## 步骤概览

1. 检查GPU与CUDA兼容性
2. 验证系统要求
3. 下载CUDA Toolkit
4. 安装CUDA Toolkit
5. 配置环境变量
6. 验证安装
7. 安装示例和文档（可选）

## 详细步骤

### 1. 检查GPU与CUDA兼容性

- 打开Windows命令提示符（CMD）或PowerShell
- 输入以下命令查看GPU信息：
  ```
  nvidia-smi
  ```
- 记下显示的驱动版本和CUDA版本信息
- 访问[NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)网站，确认你的GPU支持CUDA

### 2. 验证系统要求

- 确保你的Windows系统满足CUDA的最低要求：
  - Windows 10或Windows 11（64位）
  - Visual Studio 2019或更高版本（用于C++开发）
  - 足够的磁盘空间（至少10GB）
  - 至少4GB RAM（建议8GB或更多）

### 3. 下载CUDA Toolkit

- 访问[NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-downloads)
- 选择你的操作系统配置：
  - Operating System: Windows
  - Architecture: x86_64
  - Windows版本: 选择你的Windows版本
  - Installer Type: 选择exe（local）或网络安装
- 点击下载按钮获取安装程序

### 4. 安装CUDA Toolkit

- 运行下载的安装程序
- 选择"Express"（快速）或"Custom"（自定义）安装
  - 快速安装：自动安装所有组件
  - 自定义安装：可以选择安装位置和组件
- 在安装过程中可能会提示你安装或更新显卡驱动
  - 如果你已经安装了最新的驱动，可以取消选择驱动安装选项
- 根据向导完成安装过程

### 5. 配置环境变量

安装完成后，CUDA安装程序通常会自动设置必要的环境变量，但最好手动检查一下：

- 在Windows搜索栏中搜索"环境变量"
- 点击"编辑系统环境变量"
- 在"系统属性"窗口中点击"环境变量"按钮
- 在"系统变量"部分，检查并确保以下变量存在：
  - `CUDA_PATH`：应指向CUDA安装目录（如 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x）
  - `Path`：应包含以下路径：
    - %CUDA_PATH%\bin
    - %CUDA_PATH%\libnvvp
- 如果这些变量不存在，请手动添加

### 6. 验证安装

- 打开新的命令提示符窗口（重要，以加载新的环境变量）
- 检查CUDA编译器版本：
  ```
  nvcc --version
  ```
- 如果显示版本信息，说明安装成功

### 7. 编译CUDA示例（可选）

CUDA安装包含一系列示例程序，可以编译这些示例来验证安装：

- 找到CUDA示例目录：
  - 通常位于 C:\ProgramData\NVIDIA Corporation\CUDA Samples
- 使用Visual Studio打开示例项目或在命令行中编译
- 测试运行示例程序

## 常见问题与解决方案

### 安装失败或出错

- 确保使用管理员权限运行安装程序
- 临时禁用杀毒软件或防火墙
- 检查系统是否满足最低要求
- 下载最新的NVIDIA显卡驱动，单独安装

### nvcc命令不被识别

- 检查环境变量是否正确设置
- 重启命令提示符或PowerShell
- 重启电脑

### 无法编译CUDA程序

- 确保安装了兼容的Visual Studio版本
- 检查CUDA Toolkit与Visual Studio的兼容性
- 确认项目设置中正确配置了CUDA路径

## 多版本CUDA共存

有时你可能需要安装多个版本的CUDA。在这种情况下：

- 每个CUDA版本会安装在独立目录中
- 环境变量通常指向最后安装的版本
- 可以手动修改环境变量来切换使用的CUDA版本
- 或者在编译命令中明确指定CUDA路径

## 卸载CUDA

如果需要卸载CUDA：

- 打开Windows控制面板
- 选择"程序和功能"
- 找到并卸载"NVIDIA CUDA Toolkit"
- 同样方式卸载CUDA相关组件
- 可能需要手动清理环境变量

## 资源链接

- [CUDA文档](https://docs.nvidia.com/cuda/)
- [NVIDIA开发者论坛](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/158)
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
