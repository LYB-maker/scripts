# **Reproduction of the jax_dft Open-Source Code**

## **1、Downloading the Code：**

The code can be obtained via the [GitHub link](https://github.com/google-research/google-research/tree/master/jax_dft) provided in the [paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.126.036401) (*Kohn-Sham Equations as Regularizer: Building Prior Knowledge into Machine-Learned Physics*). Alternatively, search for the repository directly on GitHub, navigate to the project page.

Download Methods:

### (1) Create a "git here" folder locally and run the [command]:

```python
git clone https://github.com/google-research/google-research.git  
pip install -e google-research/jax_dft
```

### (2) Click "Download ZIP" in the code section of the parent directory.

For detailed steps, refer to the [Bilibili video](https://b23.tv/msw4m9R).

## **2、Code Analysis**

### Open the **jax_dft** folder with [VSCode](https://code.visualstudio.com/download), then click on '**examples**':

#### (1) **init.py**

This is a typical Python package initialization file that contains no functional code, but it plays a crucial role in maintaining legal compliance and structural integrity for open-source projects.

#### (2) **"recover_potential_from_density_and_energy.ipynb"**

This is a Python script using the JAX library for Density Functional Theory (DFT) calculations, designed to recover the potential from given electron density and energy:

##### **#Import libraries**

```python
import jax
from jax import config
import jax.numpy as jnp
from jax_dft import scf
from jax_dft import utils
import matplotlib.pyplot as plt
import numpy as np
```



##### **# Set JAX default data type to float64**

```python
config.update('jax_enable_x64', True)
```



##### **#Configure Matplotlib plotting style**

```python
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
]#颜色

def set_matplotlib_style():
  """Sets the matplotlib style for the colab notebook."""
  mpl.rcParams['image.cmap'] = 'inferno'
  # Set width and size for lines and markers.#设置线条与标志的宽度与大小
  mpl.rcParams['lines.linewidth'] = 2.5
  mpl.rcParams['lines.markersize'] = 9
  mpl.rcParams['lines.markeredgewidth'] = 0
  # Set fontsize.#设置字体大小
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



##### **#Define and plot the density and potential energy functions**

```python
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



##### **# Define grids**

```python
grids = np.linspace(-5, 5, 201)
dx = utils.get_dx(grids)
```



##### **#Define the potential energy of a quantum harmonic oscillator (QHO)**

```python
## Quantum Harmonic Oscillator

$v(x)=\frac{1}{2}k x^2$, where $k=1$.

The ground state energy is $0.5$ Hartree.
```



##### **#Use the scf.solve_noninteracting_system function to compute the energy of the non-interacting system.**

```python
qho_potential = 0.5 * grids ** 2

qho_density, qho_energy, _ = (
    scf.solve_noninteracting_system(
        qho_potential,
        num_electrons=1,
        grids=grids))
```



##### **#Print the QHO energy and plot the density versus potential**

```python
print(f'total energy: {qho_energy}')
show_density_potential(grids, qho_density, qho_potential, grey=True)
```

total energy: 0.49999993499536804

![output](C:\Users\123\Desktop\jax_dft\jax_dft\output.png)



##### **#Define the perturbation potential and compute the energy using the scf.solve_noninteracting_system function**

```python
perturbed_potential = qho_potential + np.exp(-(grids - 0.5) ** 2 / 0.04)
perturbed_density, perturbed_energy, _ = (
    scf.solve_noninteracting_system(
        perturbed_potential,
        num_electrons=1,
        grids=grids))
```



##### **#Print the perturbation energy and plot the electron density versus potential for both the original and perturbed systems**

```python
print(f'total energy: {perturbed_energy}')
_, axs = plt.subplots(nrows=2)
show_density_potential(
    grids, qho_density, qho_potential, grey=True, do_show=False, axs=axs)
show_density_potential(
    grids, perturbed_density, perturbed_potential, axs=axs)
```

total energy: 0.6339977312405621

![output-2](C:\Users\123\Desktop\jax_dft\jax_dft\output-2.png)

Add a perturbation to the potential, and it becomes evident that the electron density is no longer the original density. Next, adjust the potential based on the loss function:
$$
L = ∫f(n - nQHO)²dx + (E - EQHO)²
$$



##### **#Define the energy-density loss function**

```python
## 注意这里使用的是'jnp'而不'np'
def density_loss(output, target):
  return jnp.sum((output - target) ** 2) * dx

def energy_loss(output, target):
  return (output - target) ** 2
```



##### **#Print the density and energy loss of the current perturbed system**

```python
print(f'Current density loss {density_loss(perturbed_density, qho_density)}')
print(f'Current energy loss {energy_loss(perturbed_energy, qho_energy)}')
print(f'Current total loss {density_loss(perturbed_density, qho_density) + energy_loss(perturbed_energy, qho_energy)}')
```

Current density loss  0.014992231403549008 

Current energy loss  0.01795540939856855 

Current total loss  0.03294764080211756



##### **#Define the total loss function**

```python
def loss_fn(potential):
  density, energy, _ = scf.solve_noninteracting_system(
      potential, num_electrons=1, grids=grids)
  return density_loss(density, qho_density) + energy_loss(energy, qho_energy)
```



##### **#Print the perturbation potential energy and the QHO potential energy loss**

```python
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



##### **#Plot the gradient of the perturbed potential**

```python
plt.plot(grids, grad_fn(perturbed_potential), '--', c=COLORS[2])
plt.xlabel(r'$x$')
plt.ylabel(r'$\frac{\partial L_n}{\partial v}$')
plt.show()
```

![output-3](C:\Users\123\Desktop\jax_dft\jax_dft\output-3.png)

Now we have the gradient. Let's update the potential from the graident of loss with respect to the potential.

$$
v\leftarrow v - \epsilon\frac{\partial L}{\partial v}
$$

##### **#Optimize the potential using gradient descent and record the loss and potential at each step**

```python
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



#
