# 方法论

## 加速方法

- 合并对齐内存访问
- 足够多的并发操作数


## Q

- 1.如何确定当前是否是合并对齐内存访问？

ncu

- 2.如何确定当前的并发操作数是多少？
- 3.如何提高并发操作数？

循环展开

核函数执行配置，充分使用硬件资源

## 注意

循环展开不能过度使用，造成寄存器负担，降低性能。