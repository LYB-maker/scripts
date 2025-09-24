# vasp+vtst 计算缺陷迁移动力学（NEB）

------

原理：Eyring equation基于过渡态理论中计算双分子反应速率常数的基本方程，势垒与温度决定反应速率。
$$
 k = \frac{k_B T}{h} e^{-\frac{E_a}{RT}} 
$$
Cl-NEB（climbing image NEB，爬升图像弹性带方法）：它通过引入一个特殊的“爬升图像”来加速找到能量面上的鞍点，即过渡态。

------

## 安装

vtst：[SCRIPTS — Transition State Tools for VASP](https://theory.cm.utexas.edu/vtsttools/scripts.html)

下载、导入bin文件夹下，并赋予命令运行权限：

```python
export PATH="/home/XXX/bin/vtstscripts-1036:$PATH"
source ~/.bashrc
```

## 运行

1、结构优化

建立**ini**(初态)、**fin**(末态)文件夹，提交relax计算。

POSCAR、POTCAR、INCAR、KPOINTS、job.sh

2、

```python
dist.pl ./ini/CONTCAR ./fin/CONTCAR #一般返回值小于5Å，可继续以下步骤
nebmake.pl ./ini/CONTCAR ./fin/CONTCAR N #N为插点数，N=返回值除以0.8向下取整，插入点的算法为线性插值
cp ini/OUTCAR 00/ && cp fin/OUTCAR 0X/
nebmovie.pl 0 #0或不加参数代表用POSCAR生成xyz文件(0或不加参数时需注释原始脚本中倒数第二个if语句)；还可取1，为用CONTCAR生成
```

nebavoid.pl 1 #确保没有原子间距小于1Å的结构

![img](https://pica.zhimg.com/v2-7f41012267f1227d7c21b30ccfddb370_r.jpg)

3、在neb主文件夹下放入POTCAR，KPOINTS、job.sh(vtst编译版本)、INCAR(如下)，提交NEB计算。

```python
#NEB
[fd: CH]$ cat INCAR
SYSTEM=Cu C
ISTART=0
ICHARG=2
ENCUT=400
ISMEAR=1
SIGMA=0.2
ALGO=Fast
LREAL=Auto
ISIF=2
EDIFF=1E-05
IVDW=12
NELMIN=5

LWAVE=.F
LCHARG=.F

EDIFFG=-0.03
NSW=500

#forneb
IBRION=3
POTIM=0
IOPT=3
ICHAIN=0
LCLIMB=.TRUE
SPRING=-5
IMAGES=3
```

![img](https://pic2.zhimg.com/v2-f0de3241baa14a07a52916c429575337_r.jpg)

优化算法使用vtst的，需要如上设置IBRION=3,POTIM=0,IOPT设置见上图。（若使用vasp自带的优化器，可以使用IBRION=1或者3，不要取2；POTIM取合适的值（0.01-0.5范围内去尝试））

4、nebef.pl（后三列： [最大原子受力]、 [能量]、 [相对初态的能量]）

![image-20250922153018841](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20250922153018841.png)

（当所有插点的最大原子受力（**第二列**）都<|EDIFFG|时，计算收敛。若中间插了多个点，则所有点都要<|EDIFFG|才能收敛。）

nebbarrier.pl，生成neb.dat用于作图

![image-20250922153035454](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20250922153035454.png)

nebresults.pl （注释57到71行，否则会将OUTCAR打包）

------

## 计算细节总结：

1、初终态的POSCAR原子顺序一定要一致，dist.pl ./ini/CONTCAR ./fin/CONTCAR的返回值小于5

2、插点数为返回值除以0.8并向下取整

2、nebavoid.pl 1 #确保没有原子间距小于1Å的结构

3、nebmovie.pl  0或不加参数代表用POSCARs生成movie（**需注释掉脚本中倒数第二个if语句**）

4、在**neb主文件夹**下放入POTCAR，KPOINTS、**job.sh(vtst编译版本)**、INCAR，提交NEB计算。

5、INCAR中需修改插点数（**IMAGES**=N），job.sh中核数为插点数的**整数倍**。

5、nebresults.pl 注释57到71行，否则会将OUTCAR打包可能影响计算（解压命令：gunzip 0*/OUTCAR.gz）