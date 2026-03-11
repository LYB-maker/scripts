import numpy as np
import matplotlib.pyplot as plt

data_path = "C:/Users/123/Desktop/outputs/Symbolic_Regression/mSISSO_Matlab/formation_energy/parity_val.dat"
data = np.genfromtxt(data_path, comments='#', dtype=float)

val_true, val_pred = data[:, 0], data[:, 1]

plt.figure(figsize=(3,4), dpi=300)

colors = ['#6a3d9a']
labels = ['Validation set']

plt.scatter(val_true, val_pred, color=colors[0], label=labels[0], s=2, alpha=0.6)

lim_min = -4
lim_max = 0
plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=0.5, alpha=1.0)

plt.xlabel('True', fontsize=9, labelpad=3)
plt.ylabel('Predicted', fontsize=9, labelpad=3)
plt.xlim(lim_min, lim_max)
plt.ylim(lim_min, lim_max)

ax = plt.gca()
ax.tick_params(axis='both',which='major',direction='in')
ax.tick_params(axis='both',which='minor',direction='in')


plt.xticks(np.arange(-4.0, 0.01, 0.5), fontsize=8)
plt.yticks(np.arange(-4.0, 0.01, 0.5), fontsize=8)
plt.grid(True, which='major', linestyle='-',linewidth=0.2,alpha=0.2)
plt.grid(True, which='minor', linestyle='-',linewidth=0.2,alpha=0.1)
plt.minorticks_on()
plt.legend(loc='lower right', fontsize=6, framealpha=0)
plt.text(
    x=-3.8,          # x坐标（基于图表数据范围，0-10之间）
    y=-0.2,          # y坐标（越大概率越靠上）
    s='mSISSO',       # 要显示的文本内容
    fontsize=8,    # 字体大小
    # fontweight='bold',  # 加粗（可选）
    ha='left',      # 水平对齐方式：left/center/right
    va='top',       # 垂直对齐方式：top/center/bottom
    color='black'   # 文本颜色
    )
plt.tight_layout()
plt.savefig('mGPTIPS_B_parity_plot.jpg', dpi=300)
plt.show()