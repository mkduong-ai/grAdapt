from grAdapt.models import Sequential
import numpy as np
from utils import timer
from grAdapt.surrogate import GPR, GPRSlidingWindow, GPROnlineInsert
from grAdapt.surrogate.kernels import RationalQuadratic, RBF, Nystroem
#import warnings
#warnings.filterwarnings("error")

def rastrigin(x):
    x = np.array(x)
    return 10*len(x)+np.sum(x**2-10*np.cos(2*np.pi*x), axis=0)
    
def sphereMin(x):
    return np.sum((x)**2)

@timer
def test():
    # Kernel Tests
    # Stock Settings
    gpr = GPR()
    model = Sequential(surrogate=gpr, random_state=1)
    bounds = [(-5, 5) for i in range(2)]
    res = model.minimize(rastrigin, bounds, 50, show_progressbar=False,
                        n_random_starts = 3)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    
    # RationalQuadratic Test
    kernel = RationalQuadratic()
    gpr = GPR(kernel=kernel)
    model = Sequential(surrogate=gpr, random_state=1)
    bounds = [(-5, 5) for i in range(2)]
    res = model.minimize(rastrigin, bounds, 50, show_progressbar=False,
                        n_random_starts = 3)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    
    # RBF Test
    gpr = GPR(kernel=RBF())
    model = Sequential(surrogate=gpr, random_state=1)
    bounds = [(-5, 5) for i in range(2)]
    res = model.minimize(rastrigin, bounds, 50, show_progressbar=False,
                        n_random_starts = 3)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    
    # Nystroem Test
    gpr = GPROnlineInsert(kernel=Nystroem(RBF))
    model = Sequential(surrogate=gpr, random_state=1)
    bounds = [(-5, 5) for i in range(2)]
    res = model.minimize(rastrigin, bounds, 50, show_progressbar=False,
                        n_random_starts = 3)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    
    # Surrogate Tests
    # GPROnlineInsert
    gpr = GPROnlineInsert(kernel=RationalQuadratic())
    model = Sequential(surrogate=gpr, random_state=1)
    bounds = [(-5, 5) for i in range(2)]
    res = model.minimize(rastrigin, bounds, 100, show_progressbar=False,
                        n_random_starts = 3)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    
    # GPRSlidingWindow
    gpr = GPRSlidingWindow(kernel=RationalQuadratic(), window_size=25)
    model = Sequential(surrogate=gpr, random_state=1)
    bounds = [(-5, 5) for i in range(2)]
    res = model.minimize(rastrigin, bounds, 100, show_progressbar=False,
                        n_random_starts = 3)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
   
    

    
def main():
    test()
    try:
        test()
        print('Changing kernels \t \t Ok.')
    except:
        print('Changing kernels \t \t Not Ok.')

main()