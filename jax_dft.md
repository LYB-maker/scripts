

# **Reproduction of the jax_dft Open-Source Code**

## **1. Downloading the Code：**

The code can be obtained via the [GitHub link](https://github.com/google-research/google-research/tree/master/jax_dft) provided in the [paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.126.036401) (*Kohn-Sham Equations as Regularizer: Building Prior Knowledge into Machine-Learned Physics*). Alternatively, search for the repository directly on GitHub, navigate to the project page.

**(1) Create a "git here" folder locally and run the [command]:**

```python
git clone https://github.com/google-research/google-research.git  
pip install -e google-research/jax_dft
```

**(2) Click "Download ZIP" in the code section of the parent directory.**

For detailed steps, refer to the [Bilibili video](https://b23.tv/msw4m9R).

## **2. Code Analysis**

Open the **jax_dft** folder with [VSCode](https://code.visualstudio.com/download), then click on '**examples**':

### **(1) init.py**

This is a typical Python package initialization file that contains no functional code, but it plays a crucial role in maintaining legal compliance and structural integrity for open-source projects.

### **(2) "recover_potential_from_density_and_energy.ipynb"**

This is a Python script using the JAX library for Density Functional Theory (DFT) calculations, designed to recover the potential from given electron density and energy:

#### **Setup**

```python
# Import libraries
import jax
from jax import config
import jax.numpy as jnp
from jax_dft import scf
from jax_dft import utils
import matplotlib.pyplot as plt
import numpy as np
```

```python
# Set JAX default data type to float64
config.update('jax_enable_x64', True)
```

```python
# Configure Matplotlib plotting style
import matplotlib as mpl

COLORS = [
    '#0072b2',
    '#de8f05',
    '#009e73',
    '#cc79a7',
    '#a24f00',
    '#9467bd',
    '#56b4e9',
    '#bcbd22',
    '#7f7f7f',
]

def set_matplotlib_style():
  """Sets the matplotlib style for the colab notebook."""
  mpl.rcParams['image.cmap'] = 'inferno'
  # Set width and size for lines and markers.
  mpl.rcParams['lines.linewidth'] = 2.5
  mpl.rcParams['lines.markersize'] = 9
  mpl.rcParams['lines.markeredgewidth'] = 0
  # Set fontsize.
  mpl.rcParams['font.size'] = 18
  mpl.rcParams['axes.labelsize'] = 20
  mpl.rcParams['axes.titlesize'] = 20
  mpl.rcParams['axes.formatter.useoffset'] = False
  mpl.rcParams['legend.fontsize'] = 14
  mpl.rcParams['xtick.labelsize'] = 14
  mpl.rcParams['ytick.labelsize'] = 14
  mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS)
  mpl.rcParams['savefig.dpi'] = 120
  mpl.rcParams['savefig.bbox'] = 'tight'

set_matplotlib_style()  
```

```python
# Define and plot the density and potential energy functions
def show_density_potential(
    grids, density, potential, do_show=True, grey=False, axs=None):
  if axs is None:
    _, axs = plt.subplots(nrows=2)
  axs[0].plot(grids, density, c='0.5' if grey else COLORS[0])
  axs[1].plot(grids, potential, c='0.5' if grey else COLORS[1])
  axs[0].set_ylabel(r'$n(x)$')
  axs[1].set_ylabel(r'$v(x)$')
  axs[1].set_xlabel(r'$x$')
  if do_show:
    plt.show()
```

#### **Run**

```python
# Define grids
grids = np.linspace(-5, 5, 201)
dx = utils.get_dx(grids)
```

Quantum Harmonic Oscillator
$$
v(x)=\frac{1}{2}k x^2   (k=1)
$$
The ground state energy is 0.5 Hartree.

```python
# Use the scf.solve_noninteracting_system function to compute the energy of the non-interacting system
qho_potential = 0.5 * grids ** 2

qho_density, qho_energy, _ = (
    scf.solve_noninteracting_system(
        qho_potential,
        num_electrons=1,
        grids=grids))
```

```python
# Print the QHO energy and plot the density versus potential
print(f'total energy: {qho_energy}')
show_density_potential(grids, qho_density, qho_potential, grey=True)
```

total energy: 0.49999993499536804

![](C:\Users\123\Desktop\jax_dft\jax_dft\output\output.png)

#### Perturbed Quantum Harmonic Oscillator

```python
# Define the perturbation potential and compute the energy using the scf.solve_noninteracting_system function
perturbed_potential = qho_potential + np.exp(-(grids - 0.5) ** 2 / 0.04)
perturbed_density, perturbed_energy, _ = (
    scf.solve_noninteracting_system(
        perturbed_potential,
        num_electrons=1,
        grids=grids))
```

```python
#Print the perturbation energy and plot the electron density versus potential for both the original and perturbed systems
print(f'total energy: {perturbed_energy}')
_, axs = plt.subplots(nrows=2)
show_density_potential(
    grids, qho_density, qho_potential, grey=True, do_show=False, axs=axs)
show_density_potential(
    grids, perturbed_density, perturbed_potential, axs=axs)
```

total energy: 0.6339977312405621

![output-2](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-2.png)

Add a perturbation to the potential, and it becomes evident that the electron density is no longer the original density. Next, adjust the potential based on the loss function:
$$
L = ∫f(n - nQHO)²dx + (E - EQHO)²
$$

```python
#Define the energy-density loss function
def density_loss(output, target):
  return jnp.sum((output - target) ** 2) * dx

def energy_loss(output, target):
  return (output - target) ** 2
```

```python
# Print the density and energy loss of the current perturbed system
print(f'Current density loss {density_loss(perturbed_density, qho_density)}')
print(f'Current energy loss {energy_loss(perturbed_energy, qho_energy)}')
print(f'Current total loss {density_loss(perturbed_density, qho_density) + energy_loss(perturbed_energy, qho_energy)}')
```

Current density loss  0.014992231403549008 

Current energy loss  0.01795540939856855 

Current total loss  0.03294764080211756

```python
# Define the total loss function
def loss_fn(potential):
  density, energy, _ = scf.solve_noninteracting_system(
      potential, num_electrons=1, grids=grids)
  return density_loss(density, qho_density) + energy_loss(energy, qho_energy)
```

```python
# Print the perturbation potential energy and the QHO potential energy loss
print(f'Loss with perturbed potential {loss_fn(perturbed_potential)}')
print(f'Loss with QHO potential {loss_fn(qho_potential)}')
```

Loss with perturbed potential 0.03294764080211756 

Loss with QHO potential 0.0

##### # get the gradient of the loss function via automatic differentiation from `jax.grad`

$$
\frac{\partial L_n}{\partial v}
$$
```python
grad_fn = jax.jit(jax.grad(loss_fn))  # Compile with jit for fast grad
```

```python
# Plot the gradient of the perturbed potential
plt.plot(grids, grad_fn(perturbed_potential), '--', c=COLORS[2])
plt.xlabel(r'$x$')
plt.ylabel(r'$\frac{\partial L_n}{\partial v}$')
plt.show()
```

![output-3](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-3.png)

Now we have the gradient. Let's update the potential from the graident of loss with respect to the potential.

$$
v\leftarrow v - \epsilon\frac{\partial L}{\partial v}
$$

```python
# Optimize the potential using gradient descent and record the loss and potential at each step
potential = perturbed_potential
loss_history = []
potential_history = []
record_interval = 1000
for i in range(5001):
  if i % record_interval == 0:
    loss_value = loss_fn(potential)
    print(f'step {i}, loss {loss_value}')
    loss_history.append(loss_value)
    potential_history.append(potential)
  potential -=  30 * grad_fn(potential)
```

step 0, loss 0.03294764080211756 

step 1000, loss 1.2281985713232167 e-06 

step 2000, loss 2.699056601052081 e-07 

step 3000, loss 1.0985142405799207 e-07 

step 4000, loss 5.551165719062641 e-08 

step 5000, loss 3.074022313833837 e-08

```python
# Plot the loss as a function of optimization iterations
history_size = len(loss_history)
plt.plot(np.arange(history_size) * record_interval, loss_history)
plt.axhline(y=0, color='0.5', ls='--')
plt.xlabel('step')
plt.ylabel(r'$L$')
plt.show()
```

![output-4](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-4.png)

```python
# Plot the electron density and potential energy at each optimization step
_, axs = plt.subplots(
    nrows=2, ncols=history_size, figsize=(2.5 * history_size, 4),
    sharex=True, sharey='row')
for i, ax in enumerate(axs[0]):
  ax.plot(grids, qho_density, c='0.5')
  density, _, _ = scf.solve_noninteracting_system(
      potential_history[i], num_electrons=1, grids=grids)
  ax.plot(grids, density, '--', c=COLORS[0])
  ax.set_title(rf'$L=${loss_fn(potential_history[i]):1.1e}')

for i, ax in enumerate(axs[1]):
  ax.plot(grids, qho_potential, c='0.5')
  ax.plot(grids, potential_history[i], '--', c=COLORS[1])

# Zoom in the potential.
axs[1][0].set_xlim(-2, 2)
axs[1][0].set_ylim(0.01, 3)
axs[0][0].set_ylabel(r'$n(x)$')
axs[1][0].set_ylabel(r'$v(x)$')
plt.show()
```

![output-5](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-5.png)



```python
# Difference in electron density and potential energy before and after optimization
optimized_potential = potential_history[-1]
optimized_density, optimized_total_eigen_energies, _ = (
    scf.solve_noninteracting_system(
        optimized_potential,
        num_electrons=1,
        grids=grids))

print(f'total energy: {optimized_total_eigen_energies}')

_, axs = plt.subplots(nrows=2)
axs[0].plot(grids, optimized_density - qho_density, c=COLORS[0])
axs[0].set_ylabel(r'$\Delta n(x)$')
axs[1].plot(grids, optimized_potential - qho_potential, c=COLORS[1])
axs[1].set_ylabel(r'$\Delta v(x)$')
axs[1].set_xlabel(r'$x$')
plt.show()
```

total energy: 0.4999996749447668

![output-6](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-6.png)

```python
# Plot comparative electron density and potential energy profiles before and after optimization
_, axs = plt.subplots(nrows=2)
show_density_potential(
    grids, qho_density, qho_potential, grey=True, do_show=False, axs=axs)
show_density_potential(
    grids, optimized_density, optimized_potential, axs=axs)
```

![output-7](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-7.png)



### **（3）solve_non_interacting_system.ipynb**

This script utilizes **JAX** and a custom `jax_dft` library to solve for the ground-state electron density and total eigenenergies of a non-interacting quantum system in a diatomic chain, with integrated visualization capabilities.

#### **Installation**

```python
# For GPU runtime
pip install --upgrade jax jaxlib==0.1.62+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# Install jax-dft
git clone https://github.com/google-research/google-research.git
pip install google-research/jax_dft
```

#### **Import**

```python
import jax
from jax import config
from jax_dft import scf
from jax_dft import utils
import matplotlib.pyplot as plt
import numpy as np
```

```python
# Initialization
config.update('jax_enable_x64', True)  # Enable 64-bit precision for numerical stability  
print(f'JAX devices: {jax.devices()}')
```

#### **Run**

```python
#External Potential Construction
num_electrons = 2
grids = np.arange(-256, 257) * 0.08  # Real-space grid (-20.48 to 20.48 a.u., spacing=0.08)  
external_potential = utils.get_atomic_chain_potential(  
    grids=grids,  
    locations=np.array([-0.8, 0.8]),  # Atomic positions (symmetric)  
    nuclear_charges=np.array([1., 1.]),  # Hydrogen nuclear charges  
    interaction_fn=utils.exponential_coulomb  # Regularized Coulomb potential  
)  
```

#**Physical Model**

Diatomic system (analogous to H₂⁺) with exponential Coulomb potential to avoid singularities:

```python
V_{ext}(r) = -\sum_{i} Z_i \frac{e^{-\mu |r-R_i|}}{|r-R_i|}  
```

```python
# Non-Interacting System Solver
density, total_eigen_energies, _ = scf.solve_noninteracting_system(  
    external_potential, num_electrons=2, grids=grids)  
```

```python
# Total Energy Output
print(f'total energy: {total_eigen_energies}')  # Sum of occupied eigenenergies 
E_{total} = 2 \epsilon_0  
```

```python
# Visualization
plt.plot(grids, density, label='density')
plt.plot(grids, external_potential, label='potential')
plt.legend(loc=0)
plt.show()
```

total energy: -2.650539439312981

![output-8](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-8.png)

The **solve_non_interacting_system.py** script was enhanced by incorporating command-line argument handling and utilizing the absl library for logging, building upon the existing foundation.

### **（4）training_neural_xc_functional.ipynb**

#### **Import**

```python
import os
# Set working directory to the script's location
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_reductions'

# Configure fonts
plt.rcParams['font.sans-serif'] = ['Arial']

# Import required libraries
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
```

```python
# Checking JAX Device Information
print(f'JAX devices: {jax.devices()}')
```

print(f'JAX devices: {jax.devices()}')

#### **Load dataset**

```python
train_distances = [128, 384]  # H2 bond distances for training (units: 0.01 Bohr)
dataset = datasets.Dataset(path='h2/', num_grids=513)  # Load 1D H2 data
grids = dataset.grids  # Grid points
train_set = dataset.get_molecules(train_distances)  # Training molecules
```

```python
# Check molecular symmetry centered on grid
if not np.all(utils.location_center_at_grids_center_point(train_set.locations, grids)):
    raise ValueError('Asymmetric molecule found in training set')
```

```python
# Initialize initial density (non-self-consistent)
initial_density = scf.get_initial_density(train_set, method='noninteracting')
```

```python
# Build neural XC functional
network = neural_xc.build_global_local_conv_net(
    num_global_filters=16,
    num_local_filters=16,
    num_local_conv_layers=2,
    activation='swish',
    grids=grids,
    minval=0.1,
    maxval=2.385345,
    downsample_factor=0)
```

```python
# Add self-interaction correction
network = neural_xc.wrap_network_with_self_interaction_layer(
    network, grids=grids, interaction_fn=utils.exponential_coulomb)
```

```python
# Get neural functional calculator
init_fn, neural_xc_energy_density_fn = neural_xc.global_functional(network, grids=grids)
```

```python
# Initialize network parameters
init_params = init_fn(random.PRNGKey(0))
initial_checkpoint_index = 0
spec, flatten_init_params = np_utils.flatten(init_params)
print(f'Parameter count: {len(flatten_init_params)}')
```

number of parameters: 1568

```python
# Configure Kohn-Sham SCF parameters
num_iterations = 15  # SCF iterations
alpha = 0.5  # Density mixing factor
alpha_decay = 0.9  # Mixing decay rate
num_mixing_iterations = 1  # History steps for density mixing
density_mse_converge_tolerance = -1.  # Convergence threshold (-1 disables)
stop_gradient_step = -1  # Disable gradient stopping
```

![ks_1_column](C:\Users\123\Desktop\jax_dft\jax_dft\ks_1_column.png)

```python
# Kohn-Sham calculation core
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
```

```python
# Batch processing with vmap
_batch_jit_kohn_sham = jax.vmap(_kohn_sham, in_axes=(None, 0, 0, 0))
```

```python
# Grid integration factor
grids_integration_factor = utils.get_dx(grids) * len(grids)
```

```python
# Define loss function
def loss_fn(flatten_params, locations, nuclear_charges, initial_density, target_energy, target_density):
    states = _batch_jit_kohn_sham(flatten_params, locations, nuclear_charges, initial_density)
    loss_value = losses.trajectory_mse(target=target_energy, predict=states.total_energy[:, 10:], discount=0.9) / dataset.num_electrons
    loss_value += losses.mean_square_error(target=target_density, predict=states.density[:, -1, :]) * grids_integration_factor / dataset.num_electrons
    return loss_value
```

```python
# Automatic differentiation setup
value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
```

```python
# Checkpointing interval
save_every_n = 20
loss_record = []
```

```python
# Wrapped function for optimization
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
    print(f'Step {step}, Loss {train_set_loss}, Time {step_time}s')

    if len(loss_record) % save_every_n == 0:
        checkpoint_path = f'ckpt-{step:05d}'
        print(f'Saving checkpoint {checkpoint_path}')
        with open(checkpoint_path, 'wb') as handle:
            pickle.dump(np_utils.unflatten(spec, flatten_params), handle)

    loss_record.append(train_set_loss)
    return train_set_loss, np.array(train_set_gradient)
```

```python
# L-BFGS optimization
max_train_steps = 200
_, _, info = scipy.optimize.fmin_l_bfgs_b(
    np_value_and_grad_fn,
    x0=np.array(flatten_init_params),
    maxfun=max_train_steps,
    factr=1,
    m=20,
    pgtol=1e-14)
print(info)
```

step 0, loss 0.18682592209915364 in 18.91426110267639 sec 

Save checkpoint ckpt-00000 

step 1, loss 2.206830723734567 in 1.6032278537750244 sec 

step 2, loss 0.03004158671185065 in 1.599245309829712 sec 

step 3, loss 0.02553093700506681 in 1.601923942565918 sec 

...

step 200, loss 5.452602803517365e-05 in 1.6102423667907715 sec 

```python
# Plot training curve
plt.plot(np.minimum.accumulate(loss_record))
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Training steps')
plt.show()
plt.close()
```

![output-9](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-9.png)

#### **Visualize the model prediction on H$_2$ over training**

```python
# Visualization setup
plot_distances = [40, 56, 72, 88, 104, 120, 136, 152, 184, 200, 216, 232, 248, 264, 280,
                  312, 328, 344, 360, 376, 392, 408, 424, 456, 472, 488, 504, 520, 536, 568, 584, 600]
plot_set = dataset.get_molecules(plot_distances)
plot_initial_density = scf.get_initial_density(plot_set, method='noninteracting')
nuclear_energy = utils.get_nuclear_interaction_energy_batch(
    plot_set.locations,
    plot_set.nuclear_charges,
    interaction_fn=utils.exponential_coulomb)
```

```python
# Generalized Kohn-Sham calculator
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
```

```python
# Checkpoint loader
def get_states(ckpt_path):
    print(f'Loading checkpoint: {ckpt_path}')
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
```

```python
# Plot dissociation curves
ckpt_list = sorted(glob.glob('ckpt-?????'))
num_ckpts = len(ckpt_list)
ckpt_states = []
for ckpt_path in ckpt_list:
    ckpt_states.append(get_states(ckpt_path))
```

Load ckpt-00000 

Load ckpt-00020 

Load ckpt-00040 

Load ckpt-00060 

Load ckpt-00080 

Load ckpt-00100 

Load ckpt-00120 

Load ckpt-00140 

Load ckpt-00160 

Load ckpt-00180 

Load ckpt-00200

```python
for i, (states, ckpt_path) in enumerate(zip(ckpt_states, ckpt_list)):
    plt.plot(
        np.array(plot_distances) / 100,
        nuclear_energy + states.total_energy[:, -1],
        color=str(0.1 + 0.85 * (num_ckpts - i) / num_ckpts),
        label=ckpt_path)
```

```python
plt.plot(
    np.array(plot_distances) / 100,
    nuclear_energy + plot_set.total_energy,
    c='r', dashes=(10, 8), label='exact')
plt.xlabel(r'$R\,\,\mathrm{(Bohr)}$')
plt.ylabel(r'$E+E_\mathrm{nn}\,\,\mathsf{(Hartree)}$')
plt.legend(bbox_to_anchor=(1.4, 0.8), framealpha=0.5)
plt.show()
plt.close()
```

![output-10](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-10.png)

#### Visualize the optimal checkpoint in paper

```python
# Load optimal model
states = get_states('h2_optimal.pkl')
```

```python
# Final comparison plot
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
```

![output-11](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-11.png)

```python
# Detailed analysis for specific distance
distance_x100 = 400  # 0.01 Bohr
x_min = -10
x_max = 10

with open('h2_optimal.pkl', 'rb') as handle:
    params = pickle.load(handle)

test = dataset.get_molecules([distance_x100])

solution = kohn_sham(
    params,
    locations=test.locations[0],
    nuclear_charges=test.nuclear_charges[0])
```

```python
# Plot density evolution
_, axs = plt.subplots(
    nrows=3,
    ncols=num_iterations // 3,
    figsize=(2.5 * (num_iterations // 3), 6), sharex=True, sharey=True)
axs[2][2].set_xlabel('x')
for i, ax in enumerate(axs.ravel()):
    ax.set_title(f'Iteration {i + 1}')
    ax.plot(grids, solution.density[i], label=r'$n$')
    ax.plot(grids, test.density[0], 'k--', label=r'exact $n$')
    ax.plot(grids, solution.xc_energy_density[i], label=r'$\epsilon_\mathrm{XC}$')
    ax.set_xlim(x_min, x_max)
axs[2][-1].legend(bbox_to_anchor=(1.2, 0.8))
axs[1][0].set_ylabel('Neural XC')
plt.show()
```

```python
# Energy convergence plot
plt.plot(
    1 + np.arange(num_iterations), solution.total_energy,
    label='KS')
truth = test.total_energy[0]
plt.axhline(y=truth, ls='--', color='k', label='exact')
plt.axhspan(
    truth - 0.0016, truth + 0.0016, color='0.9', label='chemical accuracy')
plt.xlabel('SCF Iterations')
plt.ylabel('Energy')
plt.legend()
plt.show()
plt.close()
```

![output-12](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-12.png)

![output-13](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-13.png)

### Local density approximation (LDA)

```python
# LDA comparison
lda = kohn_sham(
    None,
    locations=test.locations[0],
    nuclear_charges=test.nuclear_charges[0],
    use_lda=True)
```

```python
# LDA density plot
_, axs = plt.subplots(
    nrows=3,
    ncols=num_iterations // 3,
    figsize=(2.5 * (num_iterations // 3), 6), sharex=True, sharey=True)
axs[2][2].set_xlabel('x')
for i, ax in enumerate(axs.ravel()):
    ax.set_title(f'Iteration {i + 1}')
    ax.plot(grids, lda.density[i], label=r'$n$')
    ax.plot(grids, test.density[0], 'k--', label=r'exact $n$')
    ax.plot(grids, lda.xc_energy_density[i], label=r'$\epsilon_\mathrm{XC}$')
    ax.set_xlim(x_min, x_max)
axs[2][-1].legend(bbox_to_anchor=(1.2, 0.8))
axs[1][0].set_ylabel('LDA')
plt.show()
```

```python
# LDA energy plot
plt.plot(
    1 + np.arange(num_iterations), lda.total_energy,
    label='KS')
truth = test.total_energy[0]
plt.axhline(y=truth, ls='--', color='k', label='exact')
plt.axhspan(
    truth - 0.0016, truth + 0.0016, color='0.9', label='chemical accuracy')
plt.xlabel('SCF Iterations')
plt.ylabel('Energy')
plt.legend()
plt.show()
plt.close()
[file content end]
```

![output-14](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-14.png)

![output-15](C:\Users\123\Desktop\jax_dft\jax_dft\output\output-15.png)



