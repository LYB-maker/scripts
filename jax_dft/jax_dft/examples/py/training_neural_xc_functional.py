import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
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
import matplotlib.pyplot as plt
import numpy as np
import scipy

# Set the default dtype as float64
config.update('jax_enable_x64', True)

print(f'JAX devices: {jax.devices()}')

train_distances = [128, 384]  #@param

dataset = datasets.Dataset(path='C:/Users/123/Desktop/jax_dft/jax_dft/data/h2/h2', num_grids=513)
grids = dataset.grids
train_set = dataset.get_molecules(train_distances)

#@title Check distances are symmetric
if not np.all(utils.location_center_at_grids_center_point(
    train_set.locations, grids)):
  raise ValueErrmor(
      'Training set contains examples'
      'not centered at the center of the grids.')
      
#@title Initial density
initial_density = scf.get_initial_density(train_set, method='noninteracting')

#@title Initialize network
network = neural_xc.build_global_local_conv_net(
    num_global_filters=16,
    num_local_filters=16,
    num_local_conv_layers=2,
    activation='swish',
    grids=grids,
    minval=0.1,
    maxval=2.385345,
    downsample_factor=0)
network = neural_xc.wrap_network_with_self_interaction_layer(
    network, grids=grids, interaction_fn=utils.exponential_coulomb)
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional(
    network, grids=grids)
init_params = init_fn(random.PRNGKey(0))
initial_checkpoint_index = 0
spec, flatten_init_params = np_utils.flatten(init_params)
print(f'number of parameters: {len(flatten_init_params)}')

#@markdown The number of Kohn-Sham iterations in training.
num_iterations = 15 #@param{'type': 'integer'}
#@markdown The density linear mixing factor.
alpha = 0.5 #@param{'type': 'number'}
#@markdown Decay factor of density linear mixing factor.
alpha_decay = 0.9 #@param{'type': 'number'}
#@markdown The number of density differences in the previous iterations to mix the
#@markdown density. Linear mixing is num_mixing_iterations = 1.
num_mixing_iterations = 1 #@param{'type': 'integer'}
#@markdown The stopping criteria of Kohn-Sham iteration on density.
density_mse_converge_tolerance = -1. #@param{'type': 'number'}
#@markdown Apply stop gradient on the output state of this step and all steps
#@markdown before. The first KS step is indexed as 0. Default -1, no stop gradient
#@markdown is applied.
stop_gradient_step=-1 #@param{'type': 'integer'}

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
      # The initial density of KS self-consistent calculations.
      initial_density=initial_density,
      alpha=alpha,
      alpha_decay=alpha_decay,
      enforce_reflection_symmetry=True,
      num_mixing_iterations=num_mixing_iterations,
      density_mse_converge_tolerance=density_mse_converge_tolerance,
      stop_gradient_step=stop_gradient_step)
_batch_jit_kohn_sham = jax.vmap(_kohn_sham, in_axes=(None, 0, 0, 0))

grids_integration_factor = utils.get_dx(grids) * len(grids)

def loss_fn(
    flatten_params, locations, nuclear_charges,
    initial_density, target_energy, target_density):
  """Get losses."""
  states = _batch_jit_kohn_sham(
      flatten_params, locations, nuclear_charges, initial_density)
  # Energy loss
  loss_value = losses.trajectory_mse(
      target=target_energy,
      predict=states.total_energy[
          # The starting states have larger errors. Ignore the number of 
          # starting states (here 10) in loss.
          :, 10:],
      # The discount factor in the trajectory loss.
      discount=0.9) / dataset.num_electrons
  # Density loss
  loss_value += losses.mean_square_error(
      target=target_density, predict=states.density[:, -1, :]
      ) * grids_integration_factor / dataset.num_electrons
  return loss_value

value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

# #@markdown The frequency of saving checkpoints.
# save_every_n = 20 #@param{'type': 'integer'}

# loss_record = []

# def np_value_and_grad_fn(flatten_params):
  # """Gets loss value and gradient of parameters as float and numpy array."""
  # start_time = time.time()
  # # Automatic differentiation.
  # train_set_loss, train_set_gradient = value_and_grad_fn(
      # flatten_params,
      # locations=train_set.locations,
      # nuclear_charges=train_set.nuclear_charges,
      # initial_density=initial_density,
      # target_energy=train_set.total_energy,
      # target_density=train_set.density)
  # step_time = time.time() - start_time
  # step = initial_checkpoint_index + len(loss_record)
  # print(f'step {step}, loss {train_set_loss} in {step_time} sec')

  # # Save checkpoints.
  # if len(loss_record) % save_every_n == 0:
    # checkpoint_path = f'ckpt-{step:05d}'
    # print(f'Save checkpoint {checkpoint_path}')
    # with open(checkpoint_path, 'wb') as handle:
      # pickle.dump(np_utils.unflatten(spec, flatten_params), handle)

  # loss_record.append(train_set_loss)
  # return train_set_loss, np.array(train_set_gradient)
  
# #@title Use L-BFGS optimizer to update neural network functional
# #@markdown This cell trains the model. Each step takes about 1.6s.
# max_train_steps = 200 #@param{'type': 'integer'}

# _, _, info = scipy.optimize.fmin_l_bfgs_b(
    # np_value_and_grad_fn,
    # x0=np.array(flatten_init_params),
    # # Maximum number of function evaluations.
    # maxfun=max_train_steps,
    # factr=1,
    # m=20,
    # pgtol=1e-14)
# print(info)

# #@title loss curve

# plt.plot(np.minimum.accumulate(loss_record))
# plt.yscale('log')
# plt.ylabel('loss')
# plt.xlabel('training steps')
# plt.show()

# #@title Helper functions

plot_distances = [40, 56, 72, 88, 104, 120, 136, 152, 184, 200, 216, 232, 248, 264, 280, 312, 328, 344, 360, 376, 392, 408, 424, 456, 472, 488, 504, 520, 536, 568, 584, 600] #@param
plot_set = dataset.get_molecules(plot_distances)
plot_initial_density = scf.get_initial_density(
    plot_set, method='noninteracting')
nuclear_energy = utils.get_nuclear_interaction_energy_batch(
    plot_set.locations,
    plot_set.nuclear_charges,
    interaction_fn=utils.exponential_coulomb)

def kohn_sham(
    params, locations, nuclear_charges, initial_density=None, use_lda=False):
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
    # The initial density of KS self-consistent calculations.
    initial_density=initial_density,
    alpha=alpha,
    alpha_decay=alpha_decay,
    enforce_reflection_symmetry=True,
    num_mixing_iterations=num_mixing_iterations,
    density_mse_converge_tolerance=density_mse_converge_tolerance)

def get_states(ckpt_path):
  print(f'Load {ckpt_path}')
  with open(ckpt_path, 'rb') as handle:
    params = pickle.load(handle)
  states = []
  for i in range(len(plot_distances)):
    states.append(kohn_sham(
        params,
        locations=plot_set.locations[i],
        nuclear_charges=plot_set.nuclear_charges[i],
        initial_density=plot_initial_density[i]))
  return tree_util.tree_map(lambda *x: jnp.stack(x), *states)
  
#@title Distribution of the model trained with Kohn-Sham regularizer
#@markdown Runtime ~20 minutes for 11 checkpoints.
#@markdown To speed up the calculation, you can reduce the number of
#@markdown separations to compute in
#@markdown `Helper functions -> plot_distances`

# ckpt_list = sorted(glob.glob('ckpt-?????'))
# num_ckpts = len(ckpt_list)
# ckpt_states = []
# for ckpt_path in ckpt_list:
  # ckpt_states.append(get_states(ckpt_path))
# print("States object:", states)  
# for i, (states, ckpt_path) in enumerate(zip(ckpt_states, ckpt_list)):
    # plt.plot(
    # np.array(plot_distances) / 100,
    # nuclear_energy + states.total_energy[:, -1],
    # color=str(0.1 + 0.85 * (num_ckpts - i) / num_ckpts),
    # label=ckpt_path)
    # plt.plot(
    # np.array(plot_distances) / 100,
    # nuclear_energy + plot_set.total_energy,
    # c='r', dashes=(10, 8), label='exact')
# plt.xlabel(r'$R\,\,\mathrm{(Bohr)}$')
# plt.ylabel(r'$E+E_\mathrm{nn}\,\,\mathsf{(Hartree)}$')
# plt.legend(bbox_to_anchor=(1.4, 0.8), framealpha=0.5)
# plt.show()


states = get_states('C:/Users/123/Desktop/jax_dft/jax_dft/data/h2/h2/h2_optimal.pkl')
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

distance_x100 = 400 #@param{'type': 'integer'}
#@markdown Plot range on x axis
x_min = -10 #@param{'type': 'number'}
x_max = 10 #@param{'type': 'number'}

with open('C:/Users/123/Desktop/jax_dft/jax_dft/data/h2/h2/h2_optimal.pkl', 'rb') as handle:
  params = pickle.load(handle)

test = dataset.get_molecules([distance_x100])

solution = kohn_sham(
    params,
    locations=test.locations[0],
    nuclear_charges=test.nuclear_charges[0])
    
# Density and XC energy density
_, axs = plt.subplots(
    nrows=3,
    ncols=num_iterations // 3,
    figsize=(2.5 * (num_iterations // 3), 6), sharex=True, sharey=True)
axs[2][2].set_xlabel('x')
for i, ax in enumerate(axs.ravel()):
  ax.set_title(f'KS iter {i + 1}')
  ax.plot(grids, solution.density[i], label=r'$n$')
  ax.plot(grids, test.density[0], 'k--', label=r'exact $n$')
  ax.plot(grids, solution.xc_energy_density[i], label=r'$\epsilon_\mathrm{XC}$')
  ax.set_xlim(x_min, x_max)
axs[2][-1].legend(bbox_to_anchor=(1.2, 0.8))
axs[1][0].set_ylabel('Neural XC')
plt.show()

plt.plot(
    1 + np.arange(num_iterations), solution.total_energy,
    label='KS')
truth = test.total_energy[0]
plt.axhline(y=truth, ls='--', color='k', label='exact')
plt.axhspan(
    truth - 0.0016, truth + 0.0016, color='0.9', label='chemical accuracy')
plt.xlabel('KS iterations')
plt.ylabel('Energy')
plt.legend()
plt.show()

lda = kohn_sham(
    None,
    locations=test.locations[0],
    nuclear_charges=test.nuclear_charges[0],
    use_lda=True)
    
_, axs = plt.subplots(
    nrows=3,
    ncols=num_iterations // 3,
    figsize=(2.5 * (num_iterations // 3), 6), sharex=True, sharey=True)
axs[2][2].set_xlabel('x')
for i, ax in enumerate(axs.ravel()):
  ax.set_title(f'KS iter {i + 1}')
  ax.plot(grids, lda.density[i], label=r'$n$')
  ax.plot(grids, test.density[0], 'k--', label=r'exact $n$')
  ax.plot(grids, lda.xc_energy_density[i], label=r'$\epsilon_\mathrm{XC}$')
  ax.set_xlim(x_min, x_max)
axs[2][-1].legend(bbox_to_anchor=(1.2, 0.8))
axs[1][0].set_ylabel('LDA')
plt.show()

plt.plot(
    1 + np.arange(num_iterations), lda.total_energy,
    label='KS')
truth = test.total_energy[0]
plt.axhline(y=truth, ls='--', color='k', label='exact')
plt.axhspan(
    truth - 0.0016, truth + 0.0016, color='0.9', label='chemical accuracy')
plt.xlabel('KS iterations')
plt.ylabel('Energy')
plt.legend()
plt.show()

