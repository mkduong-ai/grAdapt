from grAdapt.models import Sequential
import numpy as np
from utils import timer, enablePrint
from grAdapt.surrogate import GPR, GPRSlidingWindow, GPROnlineInsert
from grAdapt.surrogate.kernels import Nystroem, RationalQuadratic, RBF

def rastrigin(x):
    x = np.array(x)
    return 10*len(x)+np.sum(x**2-10*np.cos(2*np.pi*x), axis=0)
    
def sphereMin(x):
    return np.sum((x-1.25)**2)

@timer
def test():
    gpr = GPRSlidingWindow(kernel=Nystroem(RBF()), window_size = 10)
    model = Sequential(surrogate=gpr)
    bounds = [(-10, 10) for i in range(11)]
    res = model.minimize(sphereMin, bounds, 100, show_progressbar=False)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    print('Argmin: {xmin}'.format(xmin=x[np.argmin(y)]))
    
def main():
    try:
        test()
        print('Sliding Window \t \t \t Ok.')
    except:
        enablePrint()
        print('Sliding Window \t \t \t Not Ok.')
main()