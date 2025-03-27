import numpy as np
import matplotlib.pyplot as plt
import os

# 设置工作目录为当前脚本所在目录
os.chdir(os.path.split(os.path.realpath(__file__))[0])

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']

# 导入所需库
import glob
import pickle
import time
import jax
from jax import random
from jax import tree_util
from jax import config
import jax.numpy as jnp
from jax_dft import datasets
from jax_dft import jit_scf
from jax_dft import losses
from jax_dft import neural_xc
from jax_dft import np_utils
from jax_dft import scf
from jax_dft import utils
from jax_dft import xc
import scipy

# 设置JAX默认数据类型为float64
config.update('jax_enable_x64', True)

# 加载数据集
train_distances = [128, 384]  # 训练用的H2原子间距（单位0.01 Bohr）
dataset = datasets.Dataset(path='h2/', num_grids=513)  # 加载一维H2数据
grids = dataset.grids  # 网格点
train_set = dataset.get_molecules(train_distances)  # 获取对应距离的训练分子数据

# 检查分子是否对称居中于网格中心
if not np.all(utils.location_center_at_grids_center_point(train_set.locations, grids)):
    raise ValueError('训练集中存在不对称居中的分子')

# 初始化电子密度（非自洽）
initial_density = scf.get_initial_density(train_set, method='noninteracting')

# 构建并初始化神经网络泛函
network = neural_xc.build_global_local_conv_net(
    num_global_filters=16,
    num_local_filters=16,
    num_local_conv_layers=2,
    activation='swish',
    grids=grids,
    minval=0.1,
    maxval=2.385345,
    downsample_factor=0)

# 加入自相互作用修正层
network = neural_xc.wrap_network_with_self_interaction_layer(
    network, grids=grids, interaction_fn=utils.exponential_coulomb)

# 获取神经泛函计算函数
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional(network, grids=grids)

# 初始化网络参数
init_params = init_fn(random.PRNGKey(0))
initial_checkpoint_index = 0
spec, flatten_init_params = np_utils.flatten(init_params)
print(f'参数数量: {len(flatten_init_params)}')

# 设置Kohn-Sham自洽计算参数
num_iterations = 15  # KS迭代次数
alpha = 0.5  # 密度混合因子
alpha_decay = 0.9  # 密度混合因子的衰减因子
num_mixing_iterations = 1  # 混合历史密度的次数（1表示线性混合）
density_mse_converge_tolerance = -1.  # 密度MSE收敛阈值（-1表示禁用提前停止）
stop_gradient_step = -1  # 是否在某步之后停止梯度传播（-1表示不停止）

# 定义Kohn-Sham主函数
def _kohn_sham(flatten_params, locations, nuclear_charges, initial_density):
    return jit_scf.kohn_sham(
        locations=locations,
        nuclear_charges=nuclear_charges,
        num_electrons=dataset.num_electrons,
        num_iterations=num_iterations,
        grids=grids,
        xc_energy_density_fn=tree_util.Partial(
            neural_xc_energy_density_fn,
            params=np_utils.unflatten(spec, flatten_params)),
        interaction_fn=utils.exponential_coulomb,
        initial_density=initial_density,
        alpha=alpha,
        alpha_decay=alpha_decay,
        enforce_reflection_symmetry=True,
        num_mixing_iterations=num_mixing_iterations,
        density_mse_converge_tolerance=density_mse_converge_tolerance,
        stop_gradient_step=stop_gradient_step)

# 批量并行Kohn-Sham计算
_batch_jit_kohn_sham = jax.vmap(_kohn_sham, in_axes=(None, 0, 0, 0))

# 网格积分因子
grids_integration_factor = utils.get_dx(grids) * len(grids)

# 定义损失函数（包含能量损失和密度损失）
def loss_fn(flatten_params, locations, nuclear_charges, initial_density, target_energy, target_density):
    states = _batch_jit_kohn_sham(flatten_params, locations, nuclear_charges, initial_density)
    loss_value = losses.trajectory_mse(target=target_energy, predict=states.total_energy[:, 10:], discount=0.9) / dataset.num_electrons
    loss_value += losses.mean_square_error(target=target_density, predict=states.density[:, -1, :]) * grids_integration_factor / dataset.num_electrons
    return loss_value

# 计算损失值及其梯度（JAX自动微分）
value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

# 设置每隔多少步保存一次检查点
save_every_n = 20

loss_record = []

# 包装函数：返回损失值和梯度，并保存检查点
def np_value_and_grad_fn(flatten_params):
    start_time = time.time()
    train_set_loss, train_set_gradient = value_and_grad_fn(
        flatten_params,
        locations=train_set.locations,
        nuclear_charges=train_set.nuclear_charges,
        initial_density=initial_density,
        target_energy=train_set.total_energy,
        target_density=train_set.density)
    step_time = time.time() - start_time
    step = initial_checkpoint_index + len(loss_record)
    print(f'步骤 {step}, 损失 {train_set_loss} 用时 {step_time} 秒')

    if len(loss_record) % save_every_n == 0:
        checkpoint_path = f'ckpt-{step:05d}'
        print(f'保存检查点 {checkpoint_path}')
        with open(checkpoint_path, 'wb') as handle:
            pickle.dump(np_utils.unflatten(spec, flatten_params), handle)

    loss_record.append(train_set_loss)
    return train_set_loss, np.array(train_set_gradient)

# 使用L-BFGS优化器训练神经泛函
max_train_steps = 200  # 最大训练步数
_, _, info = scipy.optimize.fmin_l_bfgs_b(
    np_value_and_grad_fn,
    x0=np.array(flatten_init_params),
    maxfun=max_train_steps,
    factr=1,
    m=20,
    pgtol=1e-14)
print(info)

# 绘制训练损失曲线（对数坐标）
plt.plot(np.minimum.accumulate(loss_record))
plt.yscale('log')
plt.ylabel('损失')
plt.xlabel('训练步数')
plt.show()
plt.close()

# 可视化训练过程中模型在多个H2距离上的预测结果

# 设置需要绘图的H2分子间距（单位：0.01 Bohr）
plot_distances = [40, 56, 72, 88, 104, 120, 136, 152, 184, 200, 216, 232, 248, 264, 280,
                  312, 328, 344, 360, 376, 392, 408, 424, 456, 472, 488, 504, 520, 536, 568, 584, 600]
plot_set = dataset.get_molecules(plot_distances)  # 获取分子数据
plot_initial_density = scf.get_initial_density(plot_set, method='noninteracting')  # 初始密度
nuclear_energy = utils.get_nuclear_interaction_energy_batch(
    plot_set.locations,
    plot_set.nuclear_charges,
    interaction_fn=utils.exponential_coulomb)  # 计算核间相互作用能

# 定义Kohn-Sham自洽计算（可选择是否使用LDA）
def kohn_sham(params, locations, nuclear_charges, initial_density=None, use_lda=False):
    return scf.kohn_sham(
        locations=locations,
        nuclear_charges=nuclear_charges,
        num_electrons=dataset.num_electrons,
        num_iterations=num_iterations,
        grids=grids,
        xc_energy_density_fn=tree_util.Partial(
            xc.get_lda_xc_energy_density_fn() if use_lda else neural_xc_energy_density_fn,
            params=params),
        interaction_fn=utils.exponential_coulomb,
        initial_density=initial_density,
        alpha=alpha,
        alpha_decay=alpha_decay,
        enforce_reflection_symmetry=True,
        num_mixing_iterations=num_mixing_iterations,
        density_mse_converge_tolerance=density_mse_converge_tolerance)

# 加载某个训练检查点并计算多个间距下的自洽结果
def get_states(ckpt_path):
    print(f'加载检查点：{ckpt_path}')
    with open(ckpt_path, 'rb') as handle:
        params = pickle.load(handle)
    states = []
    for i in range(len(plot_distances)):
        states.append(kohn_sham(
            params,
            locations=plot_set.locations[i],
            nuclear_charges=plot_set.nuclear_charges[i],
            initial_density=plot_initial_density[i]))
    return tree_util.tree_multimap(lambda *x: jnp.stack(x), *states)

# 加载所有检查点并绘制KS总能随距离的变化（解离曲线）
ckpt_list = sorted(glob.glob('ckpt-?????'))
num_ckpts = len(ckpt_list)
ckpt_states = []
for ckpt_path in ckpt_list:
    ckpt_states.append(get_states(ckpt_path))

# 绘制所有检查点对应的解离曲线
for i, (states, ckpt_path) in enumerate(zip(ckpt_states, ckpt_list)):
    plt.plot(
        np.array(plot_distances) / 100,
        nuclear_energy + states.total_energy[:, -1],
        color=str(0.1 + 0.85 * (num_ckpts - i) / num_ckpts),
        label=ckpt_path)

# 绘制精确参考曲线
plt.plot(
    np.array(plot_distances) / 100,
    nuclear_energy + plot_set.total_energy,
    c='r', dashes=(10, 8), label='exact')
plt.xlabel(r'$R\,\,\mathrm{(Bohr)}$')  # 原子间距
plt.ylabel(r'$E+E_\mathrm{nn}\,\,\mathsf{(Hartree)}$')  # 总能+核间势能
plt.legend(bbox_to_anchor=(1.4, 0.8), framealpha=0.5)
plt.show()
plt.close()

# 加载最终训练好的最优模型进行预测
states = get_states('h2_optimal.pkl')

# 绘制最优模型的预测与真实解离曲线对比
plt.plot(
    np.array(plot_distances) / 100,
    nuclear_energy + states.total_energy[:, -1], lw=2.5, label='KSR')
plt.plot(
    np.array(plot_distances) / 100,
    nuclear_energy + plot_set.total_energy,
    c='r', dashes=(10, 8), label='exact')
plt.xlabel(r'$R\,\,\mathrm{(Bohr)}$')
plt.ylabel(r'$E+E_\mathrm{nn}\,\,\mathsf{(Hartree)}$')
plt.legend(loc=0)
plt.show()
plt.close()

# 指定某个H2间距，绘制每次KS迭代的密度与能量变化
distance_x100 = 400  # 原子间距（单位0.01 Bohr）
x_min = -10  # 横轴范围左
x_max = 10   # 横轴范围右

# 加载模型参数
with open('h2_optimal.pkl', 'rb') as handle:
    params = pickle.load(handle)

# 获取测试分子
test = dataset.get_molecules([distance_x100])

# 执行Kohn-Sham自洽计算（神经网络XC泛函）
solution = kohn_sham(
    params,
    locations=test.locations[0],
    nuclear_charges=test.nuclear_charges[0])

# 绘制每次KS迭代的密度与XC能密度
_, axs = plt.subplots(
    nrows=3,
    ncols=num_iterations // 3,
    figsize=(2.5 * (num_iterations // 3), 6), sharex=True, sharey=True)
axs[2][2].set_xlabel('x')
for i, ax in enumerate(axs.ravel()):
    ax.set_title(f'KS迭代 {i + 1}')
    ax.plot(grids, solution.density[i], label=r'$n$')
    ax.plot(grids, test.density[0], 'k--', label=r'exact $n$')
    ax.plot(grids, solution.xc_energy_density[i], label=r'$\epsilon_\mathrm{XC}$')
    ax.set_xlim(x_min, x_max)
axs[2][-1].legend(bbox_to_anchor=(1.2, 0.8))
axs[1][0].set_ylabel('Neural XC')
plt.show()

# 绘制每步KS迭代的总能量变化
plt.plot(
    1 + np.arange(num_iterations), solution.total_energy,
    label='KS')
truth = test.total_energy[0]
plt.axhline(y=truth, ls='--', color='k', label='exact')
plt.axhspan(
    truth - 0.0016, truth + 0.0016, color='0.9', label='chemical accuracy')
plt.xlabel('KS迭代次数')
plt.ylabel('能量')
plt.legend()
plt.show()
plt.close()

# 使用LDA泛函进行同样计算，用作对比
lda = kohn_sham(
    None,
    locations=test.locations[0],
    nuclear_charges=test.nuclear_charges[0],
    use_lda=True)

# 绘制LDA的密度与XC能密度演化
_, axs = plt.subplots(
    nrows=3,
    ncols=num_iterations // 3,
    figsize=(2.5 * (num_iterations // 3), 6), sharex=True, sharey=True)
axs[2][2].set_xlabel('x')
for i, ax in enumerate(axs.ravel()):
    ax.set_title(f'KS迭代 {i + 1}')
    ax.plot(grids, lda.density[i], label=r'$n$')
    ax.plot(grids, test.density[0], 'k--', label=r'exact $n$')
    ax.plot(grids, lda.xc_energy_density[i], label=r'$\epsilon_\mathrm{XC}$')
    ax.set_xlim(x_min, x_max)
axs[2][-1].legend(bbox_to_anchor=(1.2, 0.8))
axs[1][0].set_ylabel('LDA')
plt.show()

# LDA自洽能量变化
plt.plot(
    1 + np.arange(num_iterations), lda.total_energy,
    label='KS')
truth = test.total_energy[0]
plt.axhline(y=truth, ls='--', color='k', label='exact')
plt.axhspan(
    truth - 0.0016, truth + 0.0016, color='0.9', label='chemical accuracy')
plt.xlabel('KS迭代次数')
plt.ylabel('能量')
plt.legend()
plt.show()
plt.close()
