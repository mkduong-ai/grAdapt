from grAdapt.models import Sequential
import numpy as np
from utils import timer
from grAdapt.surrogate import GPR, GPRSlidingWindow, GPROnlineInsert
from grAdapt.surrogate.kernels import Nystroem, RationalQuadratic
import grAdapt.escape as esc

def rastrigin(x):
    x = np.array(x)
    return 10*len(x)+np.sum(x**2-10*np.cos(2*np.pi*x), axis=0)
    
def sphereMin(x):
    return np.sum((x-1.25)**2)

#@timer
def test():
    kernel = Nystroem(RationalQuadratic())
    gpr = GPRSlidingWindow(kernel=kernel, window_size=100)
    escape = esc.NormalDistributionDecay(gpr)
    model = Sequential(escape=escape, surrogate=gpr)
    #model = Sequential()
    bounds = [(-10, 10) for i in range(5)]
    res = model.minimize(rastrigin, bounds, 1000, show_progressbar=True)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    print('Argmin: {xmin}'.format(xmin=x[np.argmin(y)]))
    
def main():
    test()
    '''
    try:
        test()
        print('Test \t \t \t \t Ok.')
    except:
        print('Test \t \t \t \t Not Ok.')
    '''
main()