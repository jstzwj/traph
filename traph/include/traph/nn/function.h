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
	template<typename T>
	VariableInterfacePtr empty(std::initializer_list<idx_type> l, bool requires_grad = false)
	{
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

		std::shared_ptr<VariableInterface> result(new Variable<T>(dim, false));
		result->leaf_(true);

		return result;
	}

	template<typename T>
	VariableInterfacePtr zeros(std::initializer_list<idx_type> l, bool requires_grad = false)
	{
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

		std::shared_ptr<VariableInterface> result(new Variable<T>(dim, false));
		result->leaf_(true);
		std::dynamic_pointer_cast<TensorBase<T>>(result->data())->fill_(0);

		return result;
	}

	template<typename T>
	VariableInterfacePtr ones(std::initializer_list<idx_type> l, bool requires_grad = false)
	{
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

		std::shared_ptr<VariableInterface> result(new Variable<T>(dim, false));
		result->leaf_(true);
		std::dynamic_pointer_cast<TensorBase<T>>(result->data())->fill_(1);

		return result;
	}

	template<typename T>
	VariableInterfacePtr empty_like(VariableInterfacePtr input, bool requires_grad = false)
	{
		std::shared_ptr<VariableInterface> result(new Variable<T>(input->size(), false));
		result->leaf_(true);

		return result;
	}

	// arithmetic function
	VariableInterfacePtr sum(VariableInterfacePtr input)
    {
		DimVector result_dim(1);
		result_dim[0] = 1;

        VariableInterfacePtr result = input->new_empty(result_dim, true);
        std::shared_ptr<SumOp> op(new SumOp);
        if(input->requires_grad())
        {
			std::vector<VariableInterfacePtr> result_inputs { input };
            result->data_(op->forward({ input->data() }));
			result->grad_(result->data()->create_grad());
			result->grad()->fill_(0);
            result->requires_grad_(true);
            result->leaf_(false);
            result->grad_fn_(op);
            result->inputs_(result_inputs);
        }
        else
        {
            result->data_(op->forward({ input->data() }));
            result->requires_grad_(false);
            result->leaf_(false);
        }

        return result;
    }

	VariableInterfacePtr add(VariableInterfacePtr left, VariableInterfacePtr right)
	{
		DimVector result_dim;

        VariableInterfacePtr result = left->new_empty(result_dim, true);
		std::shared_ptr<AddOp> op(new AddOp);
		if (left->requires_grad() || right->requires_grad())
		{
			std::vector<VariableInterfacePtr> result_inputs{ left, right };
			result->data_(op->forward({ left->data(), right->data() }));
			result->grad_(result->data()->create_grad());
			result->grad()->fill_(0);
			result->requires_grad_(true);
			result->leaf_(false);
			result->grad_fn_(op);
			result->inputs_(result_inputs);
		}
		else
		{
			result->data_(op->forward({ left->data(), right->data() }));
			result->requires_grad_(false);
			result->leaf_(false);
		}

		return result;
	}

	VariableInterfacePtr matmul(VariableInterfacePtr left, VariableInterfacePtr right)
	{
		DimVector result_dim;

        VariableInterfacePtr result = left->new_empty(result_dim, true);
		std::shared_ptr<MatmulOp> op(new MatmulOp);
		if (left->requires_grad() || right->requires_grad())
		{
			std::vector<VariableInterfacePtr> result_inputs{ left, right };
			result->data_(op->forward({ left->data(), right->data() }));
			result->grad_(result->data()->create_grad());
			result->grad()->fill_(0);
			result->requires_grad_(true);
			result->leaf_(false);
			result->grad_fn_(op);
			result->inputs_(result_inputs);
		}
		else
		{
			result->data_(op->forward({ left->data(), right->data() }));
			result->requires_grad_(false);
			result->leaf_(false);
		}

		return result;
	}

	
	VariableInterfacePtr select(VariableInterfacePtr input, const SliceVector& slice)
	{
		DimVector result_dim;

        VariableInterfacePtr result = input->new_empty(result_dim, true);
		std::shared_ptr<SelectOp> op(new SelectOp);
		op->set_slice(slice);

		std::vector<VariableInterfacePtr> result_inputs{ input };
		result->data_(op->forward({ input->data() }));
		result->leaf_(false);

		if (input->requires_grad())
		{
			result->grad_(result->data()->create_grad());
			result->grad()->fill_(0);
			result->requires_grad_(true);
			result->grad_fn_(op);
			result->inputs_(result_inputs);
		}
		else
		{
			result->requires_grad_(false);
		}

		return result;
	}


	VariableInterfacePtr sin(VariableInterfacePtr input)
	{
		DimVector result_dim;

        VariableInterfacePtr result = input->new_empty(result_dim, true);
		std::shared_ptr<SinOp> op(new SinOp);

		std::vector<VariableInterfacePtr> result_inputs{ input };
		result->data_(op->forward({ input->data() }));
		result->leaf_(false);

		if (input->requires_grad())
		{
			result->grad_(result->data()->create_grad());
			result->grad()->fill_(0);
			result->requires_grad_(true);
			result->grad_fn_(op);
			result->inputs_(result_inputs);
		}
		else
		{
			result->requires_grad_(false);
		}

		return result;
	}

	VariableInterfacePtr transpose(VariableInterfacePtr input, idx_type dim0, idx_type dim1)
	{
		DimVector result_dim;

        VariableInterfacePtr result = input->new_empty(result_dim, true);
		std::shared_ptr<TransposeOp> op(new TransposeOp);
		op->set_dim(dim0, dim1);

		std::vector<VariableInterfacePtr> result_inputs{ input };
		result->data_(op->forward({ input->data() }));
		result->leaf_(false);

		if (input->requires_grad())
		{
			result->grad_(result->data()->create_grad());
			result->grad()->fill_(0);
			result->requires_grad_(true);
			result->grad_fn_(op);
			result->inputs_(result_inputs);
		}
		else
		{
			result->requires_grad_(false);
		}

		return result;
	}

}

#endif