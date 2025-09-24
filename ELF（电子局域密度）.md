# ELF（电子局域密度）

物理意义：找到一个与参考电子相同自旋的另一个电子的概率越小，说明这个参考电子的局域程度越高。

## 运行

1.优化结构获得CONTCAR

2.静态自洽计算：INCAR中添加参数LELF=.TRUE.

3.作图：将得到的**ELFCAR**的文件放入Vesta中

![image-20250915161814195](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20250915161814195.png)

点击Slice

![image-20250915161935276](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20250915161935276.png)

hkl决定切面 d决定切面位置

4.三种作图方法：1、二维的切面图，2、三维的等高线图，3在结构图上放置一个二维切面图（教程：[VESTA](https://www.bilibili.com/video/BV1gF411v7fD/?spm_id_from=333.337.search-card.all.click)[计算](https://www.bilibili.com/video/BV1gF411v7fD/?spm_id_from=333.337.search-card.all.click)[ELF](https://www.bilibili.com/video/BV1gF411v7fD/?spm_id_from=333.337.search-card.all.click)[局域电子密度](https://www.bilibili.com/video/BV1gF411v7fD/?spm_id_from=333.337.search-card.all.click)[_](https://www.bilibili.com/video/BV1gF411v7fD/?spm_id_from=333.337.search-card.all.click)[哔哩哔哩](https://www.bilibili.com/video/BV1gF411v7fD/?spm_id_from=333.337.search-card.all.click)[_](https://www.bilibili.com/video/BV1gF411v7fD/?spm_id_from=333.337.search-card.all.click)[bilibili](https://www.bilibili.com/video/BV1gF411v7fD/?spm_id_from=333.337.search-card.all.click)）

## 数据分析

ELF值为0~1。

ELF=1，完全局域化（红）；

ELF=0.5，说明此处电子接近于金属体系中的电子行为（黄）；

ELF=0，完全离域化（蓝）。