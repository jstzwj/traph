import pytraph.core.dtype
import pytraph.core.traph_tensor


class Storage(object):
    pass

class Tensor(object):
    def __init__(self):
        self._inner_tensor = None
    
    def __str__(self):
        if self._inner_tensor is not None:
            return self._inner_tensor.to_string()
        else:
            return "None"

class FloatTensor(Tensor):
    def __init__(self):
        self._inner_tensor = pytraph.core.traph_tensor.FloatTensor()
    

def tensor(obj):
    if type(obj) == list:
        pass
    else:
        print('unsupported obj type')

def zeros(shape):
    if type(shape) != tuple:
        raise RuntimeError('The type of shape shall be tuple.')
    
    ret = FloatTensor()

    dim = pytraph.core.traph_tensor.DimVector()
    for each in shape:
        dim.push_back(each)
    
    ret._inner_tensor = pytraph.core.traph_tensor.FloatTensor(dim)
    ret._inner_tensor.fill_(0)

    return ret
    

def ones(shape):
    if type(shape) != tuple:
        raise RuntimeError('The type of shape shall be tuple.')
    
    ret = FloatTensor()

    dim = pytraph.core.traph_tensor.DimVector()
    for each in shape:
        dim.push_back(each)
    
    ret._inner_tensor = pytraph.core.traph_tensor.FloatTensor(dim)
    ret._inner_tensor.fill_(1)

    return ret
    