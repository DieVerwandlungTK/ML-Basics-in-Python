import numpy as np

def compose(funcs):
    def _compose(f, g):
        return lambda x: f(g(x))
    
    composed = funcs[0]
    for i in range(1, len(funcs)):
        composed = _compose(funcs[i], composed)
        
    return composed

def standardize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)