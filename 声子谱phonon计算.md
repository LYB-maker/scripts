# 声子谱计算

## 一、有限位移法

1、建立2_phonon文件夹，放入POSCAR_unitcell、POTCAR、KPOINTS、INCAR

```python
mkdir 2_phonon
cp CONTCAR ./2_phonon/POSCAR-unitcell
cp POTCAR ./2_phonon/POTCAR
```

2、生成超胞

```python
phonopy -d --dim="5 5 1" -c POSCAR-unitcell #二维结构，若是块体结构，则为“5 5 5”
```

3、准备和 **POSCAR-00\*** 数目一样的文件夹，并把 **POSCAR-00\*** 复制到每个文件夹中并命名为 **POSCAR** 。每个文件夹内都需要准备 **POTCAR** 、 **POSCAR** 、 **INCAR** 、 **KPOINTS** 。并且分别计算。**INCAR** 可以设置为如下内容：

INCAR为：

```python
SYSTEM=HELLO        #随便写
PREC=High
ISTART=0
ICHARG=2
ISPIN=1
NELM= 60;NELMIN=4
NELMDL=-3
EDIFF=1E-7
ENCUT=500
IALGO=38
ADDGRID=True
LREAL=.F.
NSW=0               #注意这里和 DFPT 方法不同
IBRION=-1           #还有这里
EDIFFG=-1E-7
ISMEAR=0;SIGMA=0.01
```

4、准备一个 band.conf 文件，内容如下：

```
ATOM_NAME = Ga N 
DIM = 5 5 1
BAND = 0.0 0.0 0.0 0.333 0.333 0.0 0.5 0.0 0.0 0.0 0.0 0.0
BAND_LABELS = G K M G
FORCE_CONSTANTS = WRITE
FC_SYMMETRY = .TRUE.
```

**ATOM_NAME** ：元素名。

**DIM** ：扩胞的倍数。

**BAND** 是高对称点的坐标，三个坐标为一组，分别代表 **XYZ** 方向，一组之内每个坐标用**一个空格**隔开，而相邻两组坐标（两个高对称点坐标）用**两个空格**隔开。

**BAND_LABELS** 分别是这几个高对称点的名字。

**FORCE_CONSTANTS** ：有限位移法为**WRITE**； **DFPT** 方法中，这个参数设置为 **READ** 。

5、把每个计算目录job-*中的vasprun.xml文件复制到上级目录中，分别改名成vasprun.xml-*，*是对应的编号。

6、运行命令、数据处理

```python
phonopy -f vasprun.xml-*
phonopy --dim='5 5 1' -c POSCAR-unitcell band.conf
phonopy-bandplot --gnuplot>551.dat
```

获得二阶力常数矩阵 **FORCE_CONSTANTS** 和声子谱的数据 **551.dat**.

最后，声子谱的数据 **551.dat** 可以拖进 **[OriginLab](https://zhida.zhihu.com/search?content_id=195171608&content_type=Article&match_order=1&q=OriginLab&zhida_source=entity)** 绘图软件中绘制声子谱，**FORCE_CONSTANTS** 可以进一步作为计算**晶格热导率的“原材料”**。

## 二、DFPT密度泛函微扰理论法

1和2同上

3、提交vasp计算

```python
mkdir phonon
cp SPOSCAR ./phonon/POSCAR
```

放入：**POSCAR-unitcell** 、 **POTCAR** 、**INCAR** 、**KPOINTS**

```python
#INCAR
SYSTEM=HELLO  #随便写
PREC=High     #精度为高
ISTART=0
ICHARG=2
ISPIN=1
NELM= 60
NELMIN=4
NELMDL=-3
EDIFF=1E-7
ENCUT=500
IALGO=38
ADDGRID=True
LREAL=.F.
NSW=1          #走一个离子步
IBRION=8       #IBRION=8 就代表使用 DFPT 方法来计算声子谱
EDIFFG=-1E-7
ISMEAR=0
SIGMA=0.01
SYMPREC=1E-6

#KPOINTS
KPOINTS
0
M
2 2 1 #这里一般需要做K点的收敛性测试，资源允许的情况下可以把K点设置成 3 3 1 ，4 4 1 等再算一遍声子谱。
0 0 0 
```

4、准备一个 **band.conf** 文件

```python
ATOM_NAME = Ga N 
DIM = 5 5 1
BAND = 0.0 0.0 0.0  0.333 0.333 0.0  0.5 0.0 0.0  0.0 0.0 0.0
BAND_LABELS = G K M G
FORCE_CONSTANTS = READ
FC_SYMMETRY = .TRUE.
```

5、完成后运行

```python
phonopy --fc vasprun.xml
phonopy --dim='5 5 1' -c POSCAR-unitcell band.conf
phonopy-bandplot --gnuplot>551.dat
```

获得二阶力常数矩阵 **FORCE_CONSTANTS** 和声子谱的数据 **551.dat**。