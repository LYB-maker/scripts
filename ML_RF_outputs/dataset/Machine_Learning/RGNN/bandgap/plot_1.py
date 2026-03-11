import numpy as np
import matplotlib.pyplot as plt

# 注意：这里假设你的 train/test 文件现在都是图片这种5折格式
data_path_train = "C:/Users/123/Desktop/outputs/Machine_Learning/RGNN/bandgap/train.dat"
data_path_test = "C:/Users/123/Desktop/outputs/Machine_Learning/RGNN/bandgap/test.dat"

# 读取5折格式数据（10列：Fold1_True, Fold1_Pred, Fold2_True, Fold2_Pred, ..., Fold5_True, Fold5_Pred）
data_train = np.genfromtxt(data_path_train, comments='#', dtype=float)
data_test = np.genfromtxt(data_path_test, comments='#', dtype=float)

# 合并所有折的 True 和 Pred 列
# 训练集：把 Fold1~Fold5 的 True 列拼接，Pred 列拼接
train_true = np.concatenate([
    data_train[:, 0],   # Fold1_True
    data_train[:, 2],   # Fold2_True
    data_train[:, 4],   # Fold3_True
    data_train[:, 6],   # Fold4_True
    data_train[:, 8]    # Fold5_True
])
train_pred = np.concatenate([
    data_train[:, 1],   # Fold1_Pred
    data_train[:, 3],   # Fold2_Pred
    data_train[:, 5],   # Fold3_Pred
    data_train[:, 7],   # Fold4_Pred
    data_train[:, 9]    # Fold5_Pred
])

# 测试集：同理合并所有折
test_true = np.concatenate([
    data_test[:, 0],
    data_test[:, 2],
    data_test[:, 4],
    data_test[:, 6],
    data_test[:, 8]
])
test_pred = np.concatenate([
    data_test[:, 1],
    data_test[:, 3],
    data_test[:, 5],
    data_test[:, 7],
    data_test[:, 9]
])

# 绘图部分保持不变
plt.figure(figsize=(3, 4), dpi=300)

colors = ['#1f78b4', '#ff7f00']
labels = ['Training set', 'Test set']

plt.scatter(train_true, train_pred, color=colors[0], label=labels[0], s=2, alpha=0.6)
plt.scatter(test_true, test_pred, color=colors[1], label=labels[1], s=2, alpha=0.6)

lim_min = 0
lim_max = 10
plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k-', linewidth=0.5, alpha=1.0)

plt.xlabel('True', fontsize=9, labelpad=3)
plt.ylabel('Predicted', fontsize=9, labelpad=3)
plt.xlim(lim_min, lim_max)
plt.ylim(lim_min, lim_max)

ax = plt.gca()
ax.tick_params(axis='both',which='major',direction='in')
ax.tick_params(axis='both',which='minor',direction='in')

plt.xticks(np.arange(0.0, 10.1, 1), fontsize=8)
plt.yticks(np.arange(0.0, 10.1, 1), fontsize=8)
plt.grid(True, which='major', linestyle='-',linewidth=0.2,alpha=0.2)
plt.grid(True, which='minor', linestyle='-',linewidth=0.2,alpha=0.1)
plt.minorticks_on()
plt.legend(loc='lower right', fontsize=6, framealpha=0)
plt.text(
    x=0.4,
    y=9.6,
    s='RGNN',
    fontsize=8,
    ha='left',
    va='top',
    color='black'
)
plt.tight_layout()
# plt.savefig('RGNN_B_parity_plot.jpg', dpi=300)
plt.show()