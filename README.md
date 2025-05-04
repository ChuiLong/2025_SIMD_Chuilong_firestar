# 文件结构说明
## 对文件夹下的ARM代码文件进行说明。

其中 main.cc 文件中存储的是效率最高的代码，亦即 Barrett规约之后的代码Barrett.cc。

DITDIF_SIMD.cc文件夹是采用Montgomery规约之后结合DIT+DIF算法的代码，效率次于Barrett.cc。

Normal_SIMD.cc是采用Montgomery规约后使用常规Cooley-Tukey算法完成的代码，效率次于DITDIF_SIMD.cc。

normalntt.cc和slowmul.cc分别对应串行NTT和朴素多项式乘法，用作性能对比。

perff.txt存储了一些Profile记录。

## 对其他文件进行说明
slowsse.cpp 是SSE框架下用于测试X86系统的常规串行算法，用于加速比计算。

fastsse.cpp 是SSE框架下用于测试X86系统的Barrett规约优化SIMD并行算法，由Barrett.cc迁移而来。

Radix-4.cc 编写的四分NTT，但尚未实现串行算法。

233444及23344两个文件系上传错误，后续删除。
