@echo off
rem 设置控制台编码为GBK以正确显示中文
chcp 936

echo === 编译和运行CUDA卷积计算程序 ===

rem 检查NVCC是否在系统路径中
where nvcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到NVCC编译器。请确保已安装CUDA并添加到系统路径。
    echo 通常NVCC位于 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin
    goto :eof
)

rem 编译CUDA源代码
echo 正在编译CUDA程序...
nvcc -o convolution_cuda convolution_cuda.cu -Xcompiler "/EHsc /W3" -arch=sm_30 -fexec-charset=GBK -finput-charset=UTF-8

if %ERRORLEVEL% neq 0 (
    echo 编译失败！
    goto :eof
)

echo 编译成功！

rem 运行程序
echo 正在运行CUDA卷积计算程序...
convolution_cuda.exe

echo 程序执行完毕！

pause 