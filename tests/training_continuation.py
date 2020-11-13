from grAdapt.models import Sequential
import numpy as np
from utils import timer, enablePrint

def rastrigin(x):
    x = np.array(x)
    return 10*len(x)+np.sum(x**2-10*np.cos(2*np.pi*x), axis=0)

def sphere(x):
    return np.sum(x**2)

@timer
def test():
    model = Sequential()
    bounds = [(-10, 10) for i in range(10)]
    n = 25
    res = model.minimize(sphere, bounds, n, show_progressbar=False)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    # print('Argmin: {xmin}'.format(xmin=x[np.argmin(y)]))
    print('')
    print('Training continuation...')
    training = (x, y)
    res = model.minimize(sphere, bounds, n, show_progressbar=False)
    x = res['x']
    y = res['y']
    y_real = np.array(list(map(sphere, x)))
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    
    if np.max(np.abs(y_real-y)) >= 1e-10:
        print('Max single error: {}'.formatnp.max(np.abs(y_real-y)))
        raise Exception('y Values wrong and not equal!')
    
    if len(x) != 2*n:
        print(len(x))
        raise Exception('Training continuation FAILED.')
    
def main():
    try:
        test()
        print('Training continuation \t \t Ok.')
    except:
        enablePrint()
        print('Training continuation \t \t Not Ok.')
main()