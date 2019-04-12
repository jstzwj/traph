#ifndef TRAPH_NN_ARITHMETIC_H_
#define TRAPH_NN_ARITHMETIC_H_

#include <utility>
#include <cmath>

#include <traph/core/type.h>
#include <traph/core/index.h>
#include <traph/core/utils.h>
#include <traph/core/variable.h>
#include <traph/nn/variable.h>
#include <traph/core/tensor.h>
#include <traph/tensor/tensor.h>
#include <traph/nn/operation.h>

namespace traph
{
    template<class T>
	VariablePtr<T> sum(VariablePtr<T> input)
    {
        VariablePtr<T> result(new Variable<T>);
        std::shared_ptr<SumOp<T>> op(new SumOp<T>);
        if(input->_requires_grad)
        {
			std::vector<VariableInterfacePtr> result_inputs { std::dynamic_pointer_cast<VariableInterface>(input) };
            result->_data = op->forward({input->_data});
            result->_requires_grad = true;
            result->_leaf = false;
            result->_grad_fn = op;
            result->_inputs = result_inputs;
        }
        else
        {
            result->_data = op->forward({input->_data});
            result->_requires_grad = false;
            result->_leaf = false;
        }

        return result;
    }

}

#endif