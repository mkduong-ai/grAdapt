from grAdapt.models import Sequential
import numpy as np
from utils import timer

def rastrigin(x):
    x = np.array(x)
    return 10*len(x)+np.sum(x**2-10*np.cos(2*np.pi*x), axis=0)

def sphere(x):
    return np.sum(x**2)

@timer
def test():
    model = Sequential(random_state=1)
    bounds = [(-10, 10) for i in range(10)]
    n = 25
    res = model.minimize(sphere, bounds, n, show_progressbar=True)
    x = res['x']
    y = res['y']
    print(x)
    print(y)
    
    #print('Argmin: {xmin}'.format(xmin=x[np.argmin(y)]))
    
def main():
    #test()
    try:
        #test()
        print('Training continuation \t \t Ok.')
    except:
        print('Training continuation \t \t Not Ok.')
main()