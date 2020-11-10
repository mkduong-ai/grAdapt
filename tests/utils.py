import time
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
# decorator to measure time
def timer(function):
    def new_function():
        start_time = time.time()
        #blockPrint()
        params = function()
        elapsed = time.time() - start_time
        print('Time elapsed: {time}'.format(time=elapsed))
        #enablePrint()
        return params
    return new_function