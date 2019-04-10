#ifndef TRAPH_NN_ARITHMETIC_H_
#define TRAPH_NN_ARITHMETIC_H_

#include <utility>
#include <cmath>

#include <traph/core/type.h>
#include <traph/core/index.h>
#include <traph/core/utils.h>
#include <traph/core/variable.h>
#include <traph/tensor/tensor.h>
#include <traph/nn/variable.h>

namespace traph
{
    template<class T>
	Variable<T> abs(const Variable<T> &t)
    {
        Variable<T> result;
        // operator
        TensorBase<T> * base = t.tensor_data();
        if(t.platform() == platform_type::none)
        {
            Tensor<T> * down = dynamic_cast<Tensor<T> *>(base);
            abs(*down);
        }
        
        // record

    }

}

#endif