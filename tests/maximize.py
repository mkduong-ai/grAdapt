from grAdapt.models import Sequential
import numpy as np
from utils import timer

def sphereMax(x):
    return -np.sum(x**2)
    
def sphereMin(x):
    return np.sum(x**2)

@timer
def test1():
    model = Sequential()
    bounds = [(-10, 10) for i in range(5)]
    res = model.maximize(sphereMax, bounds, 20, show_progressbar=False)
    x = res['x']
    y = res['y']
    print('Maximum found: {ymax}'.format(ymax=np.max(y)))
    print('Argmax: {xmax}'.format(xmax=x[np.argmax(y)]))
    
    return np.max(y)

@timer
def test2():
    model = Sequential()
    bounds = [(-10, 10) for i in range(5)]
    res = model.minimize(sphereMin, bounds, 20, show_progressbar=False)
    x = res['x']
    y = res['y']
    print('Minimum found: {ymin}'.format(ymin=np.min(y)))
    print('Argmin: {xmin}'.format(xmin=x[np.argmin(y)]))
    return np.min(y)



def main():
    x1 = test1()
    x2 = test2()
    try:
        assert x1 == -x2
        print('Maximize \t \t \t Ok.')
    except:
        print('Maximize \t \t \t Not Ok.')
main()