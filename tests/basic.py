import numpy as np
from utils import timer
from grAdapt.models import Sequential
from grAdapt.surrogate import GPR
from grAdapt.surrogate.kernels import RationalQuadratic 

def rastrigin(x):
    x = np.array(x)
    return 10*len(x)+np.sum(x**2-10*np.cos(2*np.pi*x), axis=0)
    
def sphereMin(x):
    return np.sum((x-1.25)**2)

@timer
def test():
    #gpr = GPR(kernel=RationalQuadratic())
    #model = Sequential(surrogate=gpr)
    model = Sequential()
    bounds = [(-10, 10) for i in range(11)]
    res = model.minimize(rastrigin, bounds, 100, show_progressbar=True)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    print('Argmin: {xmin}'.format(xmin=x[np.argmin(y)]))
    
def main():
    try:
        test()
        print('Basic \t \t \t \t Ok.')
    except:
        print('Basic \t \t \t \t Not Ok.')

main()