import traph_tensor
import dtype

class Storage(object):
    pass

class Tensor(object):
    def __init__(self):
        self.inner_tensor = traph_tensor.tensor_f32()
    

def tensor(obj, dtype=dtype.float):
    if type(obj) == list:
        pass
    else:
        print('unsupported obj type')

def zeros(*args):
    pass

def ones(*args):
    pass