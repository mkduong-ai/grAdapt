# grAdapt: Gradient-Adaptation for Black-Box Optimization


![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen) ![Python](https://img.shields.io/badge/Python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8-yellowgreen)

**grAdapt** is a Python package for black-box optimization with fixed budget. It is running on top of scikit-learn, scipy and numpy.

It adapts first-order optimization methods to black-box functions by estimating the gradient with different approaches without the need of additional function evaluations. Further, it samples new starting points from multivariate probability distributions to escape local optima. It is a stochastic and sequential model-based optimization method (SMBO). Most SMBO techniques suffer from quadratic, cubic or even worse time complexities. This is caused by refitting the surrogate model without prior information. **grAdapt** establishes incremental learning and a sliding window technique to improve the complexity significantly. In stock settings, the runtime of **grAdapt** scales linearly with the number of function evaluations.

Instead of establishing one optimization method, **grAdapt** is a modular package where the *sampling method, surrogate, optimizer, escape function, and the domain constraint* can be changed. This makes **grAdapt** very adaptable to many optimization problems and not only specifically to black-box optimization.

Due to the fixed budget, it suits optimization problems with costly objectives. The most common application of **grAdapt** is hyperparameter optimization.

It was started in 2019 by Manh Khoi Duong as a project and was since then continually developed, maintained by the author himself under the supervision of Martha Tatusch.

## Installation

### Dependencies
grAdapt requires:
- numpy ~= 1.18
- scipy ~= 1.4
- scikit-learn ~= 0.22
- tqdm ~= 4.44
- deprecated ~= 1.2.7

### How-to install
The current stable release can be installed from the pip distribution by:
```
$ pip install grAdapt
```

The nightly release can be installed by pulling this repository, navigating to the source directory and then simply installing the `setup.py` file:
```
$ python setup.py install
```

### Testing

To verify that the installation went well without any complications, go to the source directory, then navigate to ```tests```
```
$ cd tests
```
 and run:
```
$ python run_all.py
```
All tests should end with an OK.


## First start: Optimizing the sphere function

```python
import grAdapt
from grAdapt.models import Sequential
from grAdapt.space.datatype import Integer, Float, Categorical

# Black-Box Function
def sphereMin(x):
    return np.sum(x**2)

# create model to optimize
model = Sequential()

# defining search space
var1 = Float(low=-10, high=10)
var2 = Float(low=-10, high=10)
var3 = Float(low=-10, high=10)
bounds = [var1, var2, var3]

# minimize
n_evals = 100 # budget/number of function evaluations
res = model.minimize(sphereMin, bounds, n_evals)

# getting the history
x = res['x']
y = res['y']

# best solutions
x_sol = res['x_sol']
y_sol = res['y_sol']
```

## Citation

When using **grAdapt** in your publication, we would appreciate if you cite us.

BibTeX entry:
```
@misc{grAdapt,
title={grAdapt: Gradient-Adaptation for Black-Box Optimization},
author={Manh Khoi Duong and Martha Tatusch and Stefan Conrad and Gunnar W. Klau}
howpublished={The Python Package Index: PyPi}
}
```
## License

This project is distributed under the Apache License 2.0.
