#ifndef TRAPH_NN_VARIABLE_H_
#define TRAPH_NN_VARIABLE_H_

#include <traph/core/tensor.h>

namespace traph
{
    template<class T>
    class Variable
    {
    private:
        TensorBase<T> _data;
        TensorBase<T> _grad;
    };
}


#endif