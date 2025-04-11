@echo off
echo Setting up display environment...
REM 设置控制台代码页为简体中文 (GB2312/GBK)

chcp 936
echo Console code page set to Chinese Simplified (936)
REM 控制台代码页已设置为简体中文

REM 创建一个临时宏来编译程序使用GBK编码
echo Modifying source file encoding...

REM 保存当前目录
set CURRENT_DIR=%CD%

REM 编译和运行各个程序
echo.
echo Compiling and running programs...
echo.

REM 编译所有程序
g++ -o convolution_parallel -fexec-charset=GBK -finput-charset=UTF-8 convolution_cpp_parallel.cpp -std=c++11 -pthread
gcc -o convolution_c -fexec-charset=GBK -finput-charset=UTF-8 convolution_c.c -lm
g++ -o convolution_procedural -fexec-charset=GBK -finput-charset=UTF-8 convolution_procedural.cpp -std=c++11

REM 运行程序
echo.
echo Running C++ parallel version...
echo ==================================== 
convolution_parallel.exe
echo ====================================
echo.

echo Running C language version...
echo ====================================
convolution_c.exe
echo ====================================
echo.

echo Running procedural C++ version...
echo ====================================
convolution_procedural.exe
echo ====================================

echo.
echo All programs have been executed!
pause 