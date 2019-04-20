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

    def __getitem__(self, given):
        slice_vector = pytraph.core.traph_tensor.SliceVector()
        if isinstance(given, slice):
            slice_vector.push_back(pytraph.core.traph_tensor.Slice(given.start, given.step, given.stop))
        elif isinstance(given, tuple):
            for each_slice in given:
                if isinstance(given, slice):
                    slice_vector.push_back(pytraph.core.traph_tensor.Slice(each_slice.start, each_slice.step, each_slice.stop))
                else:
                    slice_vector.push_back(pytraph.core.traph_tensor.Slice(each_slice, 1, each_slice+1))
        else:
            slice_vector.push_back(pytraph.core.traph_tensor.Slice(given, 1, given+1))

        return self._inner_tensor.select(slice_vector)

    def __setitem__(self,key,value):
        self.dict[key] = value

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
    