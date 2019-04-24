#ifndef TRAPH_NN_FUNCTION_H_
#define TRAPH_NN_FUNCTION_H_

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
	// creation function
	template<class T>
	VariablePtr<T> zeros(std::initializer_list<idx_type> l, bool requires_grad = false)
	{
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

		std::shared_ptr<Variable<T>> result(new Variable<T>(dim, false));
		result->leaf_(true);
		result->fill_(0);

		return result;
	}

	template<class T>
	VariablePtr<T> ones(std::initializer_list<idx_type> l, bool requires_grad = false)
	{
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

		std::shared_ptr<Variable<T>> result(new Variable<T>(dim, false));
		result->leaf_(true);
		result->fill_(1);

		return result;
	}


	// arithmetic function
    template<class T>
	VariablePtr<T> sum(VariablePtr<T> input)
    {
        VariablePtr<T> result(new Variable<T>);
        std::shared_ptr<SumOp> op(new SumOp);
        if(input->_requires_grad)
        {
			std::vector<VariableInterfacePtr> result_inputs { std::dynamic_pointer_cast<VariableInterface>(input) };
            result->_data = std::dynamic_pointer_cast<TensorBase<T>>(op->forward({ input->_data }));
			result->_grad = result->_data->create_grad();
            result->_requires_grad = true;
            result->_leaf = false;
            result->_grad_fn = op;
            result->_inputs = result_inputs;
        }
        else
        {
            result->_data = std::dynamic_pointer_cast<TensorBase<T>>(op->forward({ input->_data }));
            result->_requires_grad = false;
            result->_leaf = false;
        }

        return result;
    }

	template<class T>
	VariablePtr<T> add(VariablePtr<T> left, VariablePtr<T> right)
	{
		VariablePtr<T> result(new Variable<T>);
		std::shared_ptr<AddOp> op(new AddOp);
		if (left->_requires_grad || right->_requires_grad)
		{
			std::vector<VariableInterfacePtr> result_inputs{ left, right };
			result->_data = std::dynamic_pointer_cast<TensorBase<T>>(op->forward({ left->_data, right->_data }));
			result->_grad = result->_data->create_grad();
			result->_grad->fill_(0);
			result->_requires_grad = true;
			result->_leaf = false;
			result->_grad_fn = op;
			result->_inputs = result_inputs;
		}
		else
		{
			result->_data = std::dynamic_pointer_cast<TensorBase<T>>(op->forward({ left->_data, right->_data }));
			result->_requires_grad = false;
			result->_leaf = false;
		}

		return result;
	}

	template<class T>
	VariablePtr<T> matmul(VariablePtr<T> left, VariablePtr<T> right)
	{
		VariablePtr<T> result(new Variable<T>);
		std::shared_ptr<MatmulOp> op(new MatmulOp);
		if (left->_requires_grad || right->_requires_grad)
		{
			std::vector<VariableInterfacePtr> result_inputs{ left, right };
			result->_data = std::dynamic_pointer_cast<TensorBase<T>>(op->forward({ left->_data, right->_data }));
			result->_grad = result->_data->create_grad();
			result->_grad->fill_(0);
			result->_requires_grad = true;
			result->_leaf = false;
			result->_grad_fn = op;
			result->_inputs = result_inputs;
		}
		else
		{
			result->_data = std::dynamic_pointer_cast<TensorBase<T>>(op->forward({ left->_data, right->_data }));
			result->_requires_grad = false;
			result->_leaf = false;
		}

		return result;
	}

	
	template<class T>
	VariablePtr<T> select(VariablePtr<T> input, const SliceVector& slice)
	{
		VariablePtr<T> result(new Variable<T>);
		std::shared_ptr<SelectOp> op(new SelectOp);
		op->set_slice(slice);

		std::vector<VariableInterfacePtr> result_inputs{ input };
		result->_data = std::dynamic_pointer_cast<TensorBase<T>>(op->forward({ input->_data }));
		result->_leaf = false;

		if (input->requires_grad())
		{
			result->_grad = result->_data->create_grad();
			result->_grad->fill_(0);
			result->_requires_grad = true;
			result->_grad_fn = op;
			result->_inputs = result_inputs;
		}
		else
		{
			result->_requires_grad = false;
		}

		return result;
	}


	template<class T>
	VariablePtr<T> sin(VariablePtr<T> input)
	{
		VariablePtr<T> result(new Variable<T>);
		std::shared_ptr<SinOp> op(new SinOp);

		std::vector<VariableInterfacePtr> result_inputs{ input };
		result->_data = std::dynamic_pointer_cast<TensorBase<T>>(op->forward({ input->_data }));
		result->_leaf = false;

		if (input->requires_grad())
		{
			result->_grad = result->_data->create_grad();
			result->_grad->fill_(0);
			result->_requires_grad = true;
			result->_grad_fn = op;
			result->_inputs = result_inputs;
		}
		else
		{
			result->_requires_grad = false;
		}

		return result;
	}

}

#endif