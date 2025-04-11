@echo off
chcp 65001
echo 正在编译并运行卷积计算程序...

echo.
echo 第一步：编译程序...
echo.

echo 编译 C++ 多核并行版本卷积程序...
g++ -o convolution_parallel convolution_cpp_parallel.cpp -std=c++11 -pthread

echo 编译 C 语言版本卷积程序...
gcc -o convolution_c convolution_c.c -lm

echo 编译面向过程 C++ 版本卷积程序...
g++ -o convolution_procedural convolution_procedural.cpp -std=c++11

echo.
echo 第二步：运行程序...
echo.

echo 运行 C++ 多核并行版本...
echo ====================================
convolution_parallel.exe
echo ====================================
echo.

echo 运行 C 语言版本...
echo ====================================
convolution_c.exe
echo ====================================
echo.

echo 运行面向过程 C++ 版本...
echo ====================================
convolution_procedural.exe
echo ====================================
echo.

echo 程序运行完成！
pause 