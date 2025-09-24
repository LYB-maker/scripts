# VASP：使用Lobster计算COHP

Lobster是一款用于计算晶体轨道哈密顿布居（COHP）的工具。COHP是一种分析化学键的方法，通过计算轨道间的相互作用来理解化学键的性质。以下是使用Lobster进行COHP计算的步骤和注意事项。

------

## 运行

结构优化后，在COHP文件夹准备以下文件：

INCAR KPOINTS job.sh POSCAR(结构优化后的) POTCAR

```python
head POSCAR #查看POSCAR结构
grep PAW POTCAR #确保使用PAW赝势
使用vasp_std版本
grep NBANDS OUTCAR(结构优化) #确定NBANDS
```

```python
#INCAR：
ISTART  = 0            
ICHARG  = 2            
ISYM    =-1      # 这个一定要用-1或者0，不能处理k点对称性，关闭对称性         
ENCUT   = xxxxxx # 自己进行收敛性测试获得合适的ENCUT          
PREC    = Accurate     
ISMEAR  = -5     # 如果你这里用的是1或者0，那么你的lobsterin就要相应的增加一个开关：  gaussianSmearingWidth （建议先使用0，在后面lobsterin中设置展宽值ISGMA）    
NELM    = 100          
NELMIN  = 2            
EDIFF   = 1e-8           
IBRION  = -1           
ISIF    = 0  
NSW     = 0       #单点计算     
ISPIN   = 1       # 你要是算自旋，就用2  
LREAL   = .F.      
NBANDS  = 200     #NBANDS>NELECT for COHP, 1.5 or 1.7*NBANDS 
LWAVE   = .T.     #delete old WAVECAR  
NPAR    = 4            
NEDOS   = 2000            
LORBIT  = 12      # 尽量别用11了，我看官方给的例子里面的INCAR都是12
```

**NBANDS**: 设置为总价电子数的1.5到1.7倍。**NSW**: 设置为0，进行单点计算。**LWAVE**: 设置为T，删除旧的WAVECAR文件。**ISYM**: 设置为-1，关闭对称性。**LORBIT**: 设置为11，用于DOS计算。**NEDOS**: 设置为2000，用于DOS计算。

提交单电能计算。

------

**python get_bond_total.py**得diatances.dat

```python
#!/usr/bin/env python
import os
import sys
from pprint import pprint
from collections import defaultdict
from itertools import combinations, product

import numpy as np
from pymatgen.core.structure import Structure

info="""使用说明:
python get_bond_total.py
不需要输入任何参数 
"""
print(info)

struct = Structure.from_file("POSCAR")

# 获得指定的两组元素的坐标
sites_custom1 = struct.sites
index_custom1 = [idx+1 for idx, site in enumerate(struct.sites)] # 因为python的索引编号是从0开始的

pairs = list(combinations(sites_custom1, r=2))
idxs = list(combinations(index_custom1, r=2))
# 获得每个原子对的距离并且存储
pairs_dist = defaultdict(list)
for pair, idx in zip(pairs, idxs):
    d = struct.lattice.get_all_distances(pair[0].frac_coords, pair[1].frac_coords)[0][0]
    pair_name = str(pair[0].specie) + str(idx[0]) + '-' + str(pair[1].specie) + str(idx[1])
    pairs_dist[np.round(d, decimals=3)].append(pair_name)
pairs_dist = sorted(pairs_dist.items())

# 打印相关信息
print("Note: --------------------")
print("{} {}".format("Number", "Elements"))
for idx, site in enumerate(struct.sites):
    print("{:<8} {:<8}".format(idx+1, site.specie))

print("Note: --------------------")
print("{} {} {}".format("Species", "Species", "distance"))
for d, pairs in pairs_dist:
    print("distance = {}, Number={}".format(d, len(pairs)))
    print("    "+"   ".join(pairs))

with open("distance.dat", "w") as f:
    for d, pairs in pairs_dist:
        f.write("distance = {}, Number={}\n".format(d, len(pairs)))
        f.write("    "+"   ".join(pairs))
        f.write("\n")
```

**losterin输入文件**

```python 
COHPstartEnergy  -10
COHPendEnergy    5
usebasisset pbeVaspFit2015 #基组
#每种元素基函数. 程序可以自动指定，但是建议自己手动指定，通常从PAW赝势文件的电子构型开始是一个好的选择，如
basisfunctions Ce 4f 5d 6s
basisfunctions Nb 4d 5s
basisfunctions N 2s 2p
cohpGenerator from 1.971 to 2.223 type Nb type N
#指定COHP分析的原子对，如果对单个轨道COHP感兴趣，添加orbitalwise关键词
#cohpbetween atom 1 atom 10 (orbitalwise)
#gaussianSmearingWidth 0.05
skipDOS
skipCOOP
saveProjectionToFile
loadProjectionFromFile
```

**COHPstartEnergy** 和 **COHPendEnergy**: 设置COHP输出的能量范围。

**usebasisset**: 指定基组。如果不设置(空行)就是默认基组，如果要覆盖默认基集，可使用此关键字进行覆盖。它目前支持bunge、koga和pbeVaspFit2015。

**basisfunctions**: 指定每种元素的价电子轨道。

**cohpGenerator**: 指定计算的原子对及其距离范围。

**gaussianSmearingWidth**: 设置高斯展宽(INCAR中ISMEAR  = 0时需要设置，展宽默认0.05)。

**skipDOS** 和 **skipCOOP**: 跳过DOS和COOP计算。

**saveProjectionToFile** 和 **loadProjectionFromFile**: 保存和读取投影计算结果。

提交lobster计算：lobster-5.1.1(运行命令？？？)

在计算结束后，注意检查spilling，**spilling值越低**，计算结果越准确。（最低<10%）

------

（**python get_cohpfile.py**）脚本运行报错

```
#!/usr/bin/env python
import os
import sys
import argparse
from pymatgen.core.structure import Structure

# 设置命令行参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="生成LOBSTER输入文件")
    parser.add_argument('-e', '--COHPEnergy', nargs="+", type=float, help="COHP起始终止能量")
    parser.add_argument('-s', '--species_custom', nargs="+", type=str, help="元素1, 2")
    parser.add_argument('-z', '--zvalances', nargs="+", type=str, help="元素1, 2 及其价电子轨道, 例如 -z Nb:4s,4p,4d,5s H:1s")
    parser.add_argument('-d', '--d_limit', nargs="+", type=float, help="原子间最小最大距离")
    parser.add_argument('-m', '--mode', type=int, required=True, help="选择生成的输入文件版本: 5 或 0")
    return parser.parse_args()

# 解析价电子轨道参数 '-z'
def parse_zvalances(zvalances):
    zval_dict = {}
    for entry in zvalances:
        element, orbitals = entry.split(":")
        zval_dict[element] = ' '.join(orbitals.split(","))
    return zval_dict

# 写入lobsterin文件
def write_lobsterin(dirname, mode=5, COHPstartEnergy=None, COHPendEnergy=None, species_custom1=None, species_custom2=None, lower_d=None, upper_d=None, struct=None, zval_dict=None):
    if mode == 5:
        lobsterin_path = os.path.join(dirname, "lobsterin")
        with open(lobsterin_path, "w") as f:
            f.write('COHPstartEnergy  {}\n'.format(COHPstartEnergy))
            f.write('COHPendEnergy    {}\n'.format(COHPendEnergy))
            f.write('usebasisset pbeVaspFit2015\n')  # 基组
            for spe in struct.types_of_specie:
                f.write('basisfunctions {} {}\n'.format(spe.name, zval_dict[spe.name]))   # 基组（直接使用根据vasp拟合的基组以及默认的基函数）
            # 这里会出现一个非常严重的计算问题！！！！！！！！！！！！！！
            # lobster认为你指定的原子对是不具有周期性的
            # 你用pymatgen脚本找到的距离是包含周期性的，把这原子对输入给lobsterin
            # 它认不出来这个距离是周期性的，它会按照原胞内的距离考虑两个原子的成键。
            # 所以这里我抛弃了设置原子对来计算成键强度的方法。
            # 改用设置键长来获得原子对，lobster有自己的算法来获得原子对。
            # for pair, idx, d in pairs_idxs_d:
            #     if lower_d <= d <= upper_d:
            #         f.write("cohpbetween atom {} and atom {}\n".format(idx[0], idx[1]))
            f.write("cohpGenerator from {} to {} type {} type {}\n".format(lower_d, upper_d, species_custom1, species_custom2))
    elif mode == 0:
        lobsterin_path = os.path.join(dirname, "lobsterin")
        with open(lobsterin_path, "w") as f:
            f.write('COHPstartEnergy  {}\n'.format(COHPstartEnergy)) 
            f.write('COHPendEnergy    {}\n'.format(COHPendEnergy))
            f.write('usebasisset pbeVaspFit2015\n')  #  # 基组（直接使用根据vasp拟合的基组以及默认的基函数）
            f.write('gaussianSmearingWidth 0.05\n')
            for spe in struct.types_of_specie:
                f.write('basisfunctions {} {}\n'.format(spe.name, zval_dict[spe.name]))   # 基组（直接使用根据vasp拟合的基组以及默认的基函数）
            # 这里会出现一个非常严重的计算问题！！！！！！！！！！！！！！
            # lobster认为你指定的原子对是不具有周期性的
            # 你用pymatgen脚本找到的距离是包含周期性的，把这原子对输入给lobsterin
            # 它认不出来这个距离是周期性的，它会按照原胞内的距离考虑两个原子的成键。
            # 所以这里我抛弃了设置原子对来计算成键强度的方法。
            # 改用设置键长来获得原子对，lobster有自己的算法来获得原子对。
            # for pair, idx, d in pairs_idxs_d:
            #     if lower_d <= d <= upper_d:
            #         f.write("cohpbetween atom {} and atom {}\n".format(idx[0], idx[1]))
            f.write("cohpGenerator from {} to {} type {} type {}\n".format(lower_d, upper_d, species_custom1, species_custom2))
    else:
        print(f"mode = {mode}, The scripts doesn't support it")
# 主函数
def main():
    # 获取命令行参数
    args = parse_args()

    # 提取参数
    COHPstartEnergy = args.COHPEnergy[0]
    COHPendEnergy = args.COHPEnergy[1]
    species_custom1 = args.species_custom[0]
    species_custom2 = args.species_custom[1]
    lower_d = args.d_limit[0]
    upper_d = args.d_limit[1]
    mode = args.mode  # 选择生成的输入文件版本

    # 解析价电子轨道
    zval_custom = parse_zvalances(args.zvalances)

    # 输出解析后的价电子信息（可选）
    print("Parsed Z-Valences:")
    for element, orbitals in zval_custom.items():
        print(f"{element}: {orbitals}")

    # 读取结构文件
    struct = Structure.from_file("POSCAR")

    # 创建输出目录
    dirs = "{}_{}_{}_{}".format(species_custom1, species_custom2, lower_d, upper_d)
    if not os.path.exists(dirs):
        os.mkdir(dirs)

    # 创建符号链接
    files = ['WAVECAR', 'CONTCAR', 'KPOINTS', 'OUTCAR', 'POTCAR', 'vasprun.xml']
    for file in files:
        if os.path.exists(file):
            os.system(f"ln -s {os.path.abspath(file)} {dirs}")
        else:
            print(f"{file} does not exist.")

    # 写入LOBSTER输入文件
    write_lobsterin(dirs, mode=mode, COHPstartEnergy=COHPstartEnergy, COHPendEnergy=COHPendEnergy, species_custom1=species_custom1, species_custom2=species_custom2, lower_d=lower_d, upper_d=upper_d, struct=struct, zval_dict=zval_custom)

    # 输出信息
    info = """Note: -------------------------------
1. 如果运行了lobster得不到COHP，说明NBANDS不够多
2. 如果运行了vasp发现不收敛，可以尝试做如下改变
    ISTART  = 0   ---->  ISTART  = 1   读取波函数
    ICHARG  = 2   ---->  ICHARG  = 11  读取电荷密度
"""
    print(info)

# 确保仅在脚本作为主程序运行时才执行
if __name__ == "__main__":
    main()
```

------

## Lobster的输出文件包括：

**ICOHPLIST.lobster**: 包含积分晶体轨道哈密顿布居。

**CHARGE.lobster**: 包含Mulliken电荷和Loewdin电荷。

**ICOBILIST.lobster** 和 **COBICAR.lobster**: 包含共价键指标。

**MadelungEnergies.lobster** 和 **SitePotentials.lobster**: 包含离子键指标。

COHPCAR.lobster 文件包含COHP结果：

![img](https://pic4.zhimg.com/v2-94d84fbfc746650111eb1bf0023381a5_r.jpg)

第一列为能量，第二列为pCOHP，将第二列乘-1可得-pCOHP，使用origin作能量与-pCOHP的图像。



计算出较低charge spilling值的方法：

```python
ISTART  = 0            
ICHARG  = 2            
ISYM    = -1              
PREC    = Accurate     
ISMEAR  = 0    
NELM    = 100          
NELMIN  = 2            
EDIFF   = -0.01
ENCUT   = 500
EDIFF = 1e-8           
IBRION  = -1           
ISIF    = 0  
NSW     = 0            
ISPIN   = 1      
LREAL   = .F.      
NBANDS  = 200     
LWAVE   = .TRUE.      
NPAR    = 4            
NEDOS   = 2000            
LORBIT  = 12      

#hse06
LHFCALC = .T.   #hse
HFSCREEN = 0.2   #hse06
ALGO = Damped
```

```python
COHPstartEnergy  XX
COHPendEnergy    XX
usebasisset pbeVaspFit2015
basisfunctions XX 价电子 
basisfunctions XX 价电子
basisfunctions XX 价电子
cohpGenerator from 最小键长 to 最大键长 type XX type XX
gaussianSmearingWidth 0.05
skipDOS
skipCOOP
saveProjectionToFile
loadProjectionFromFile
```

先运行lobster-5.1.1找到lobsterout文件中的**recommended basis functions: XXX**，复制并替换lobsterin基组，重新运行lobsterin。











 【VASP基础03】VASP+LOBSTER计算晶体轨道布局COHP定量衡量成键强度 - 知乎](https://zhuanlan.zhihu.com/p/668003243)
