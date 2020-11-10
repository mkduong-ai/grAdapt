from grAdapt.models import Sequential
import numpy as np
from utils import timer
from grAdapt.surrogate import GPR
from grAdapt.surrogate.kernels import RationalQuadratic 

def rastrigin(x):
    x = np.array(x)
    return 10*len(x)+np.sum(x**2-10*np.cos(2*np.pi*x), axis=0)
    
def sphereMin(x):
    return np.sum(x**2)

@timer
def test():
    #gpr = GPR(kernel=RationalQuadratic())
    #model = Sequential(surrogate=gpr)
    model = Sequential()
    bounds = [(-10, 10) for i in range(5)]
    res = model.minimize(rastrigin, bounds, 10, show_progressbar=False,
                         auto_checkpoint=True)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    print('Argmin: {xmin}'.format(xmin=x[np.argmin(y)]))
    res2 = model.load_checkpoint(model.checkpoint_file)
    
    assert (res2['x'] == res['x']).all()
    assert (res2['y'] == res['y']).all()
    assert (res2['x_sol'] == res['x_sol']).all()
    
def main():
    try:
        test()
        print('Loading Checkpoint \t \t Ok.')
    except:
        print('Loading Checkpoint \t \t Not Ok.')

main()