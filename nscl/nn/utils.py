import numpy as np
import torch

def get_memory(model,_print_=True):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    if _print_:
        print(model)
        print('Model {} : params: {:4f}M'.format(model.__class__.__name__, para * 4 / 1024 / 1024))
    return para