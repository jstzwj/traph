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

#define UNARY_OP(name, op_name)                                           \
	VariableInterfacePtr name(VariableInterfacePtr input)                 \
	{                                                                     \
		DimVector result_dim;                                             \
        VariableInterfacePtr result = input->new_empty(result_dim, true); \
		std::shared_ptr<op_name> op(new op_name);                         \
		std::vector<VariableInterfacePtr> result_inputs{ input };         \
		result->data_(op->forward({ input->data() }));                    \
		if (input->requires_grad())                                       \
		{                                                                 \
			result->grad_(result->data()->create_grad());                 \
			result->grad()->fill_(0);                                     \
			result->requires_grad_(true);                                 \
			result->grad_fn_(op);                                         \
			result->inputs_(result_inputs);                               \
		}                                                                 \
		else                                                              \
		{                                                                 \
			result->requires_grad_(false);                                \
		}                                                                 \
		return result;                                                    \
	}

#define BINARY_OP(name, op_name)                                                           \
	VariableInterfacePtr name(VariableInterfacePtr left, VariableInterfacePtr right)       \
	{                                                                                      \
		DimVector result_dim;                                                              \
        VariableInterfacePtr result = left->new_empty(result_dim, true);                   \
		std::shared_ptr<op_name> op(new op_name);                                          \
		result->data_(op->forward({ left->data(), right->data() }));                       \
		if (left->requires_grad() || right->requires_grad())                               \
		{                                                                                  \
			std::vector<VariableInterfacePtr> result_inputs{ left, right };                \
			result->grad_(result->data()->create_grad());                                  \
			result->grad()->fill_(0);                                                      \
			result->requires_grad_(true);                                                  \
			result->grad_fn_(op);                                                          \
			result->inputs_(result_inputs);                                                \
		}                                                                                  \
		else                                                                               \
		{                                                                                  \
			result->requires_grad_(false);                                                 \
		}                                                                                  \
		return result;                                                                     \
	}


	// creation function
	template<typename T>
	VariableInterfacePtr empty(std::initializer_list<idx_type> l, bool requires_grad = false)
	{
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

		std::shared_ptr<VariableInterface> result(new Variable<T>(dim, false));

		return result;
	}

	template<typename T>
	VariableInterfacePtr zeros(std::initializer_list<idx_type> l, bool requires_grad = false)
	{
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

		std::shared_ptr<VariableInterface> result(new Variable<T>(dim));
		std::dynamic_pointer_cast<TensorBase<T>>(result->data())->fill_(0);

		return result;
	}

	template<typename T>
	VariableInterfacePtr ones(std::initializer_list<idx_type> l, bool requires_grad = false)
	{
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

		std::shared_ptr<VariableInterface> result(new Variable<T>(dim));
		std::dynamic_pointer_cast<TensorBase<T>>(result->data())->fill_(1);

		return result;
	}

	template<typename T>
	VariableInterfacePtr empty_like(VariableInterfacePtr input, bool requires_grad = false)
	{
		std::shared_ptr<VariableInterface> result(new Variable<T>(input->size(), false));

		return result;
	}

	// arithmetic function
	UNARY_OP(sum, SumOp)

	BINARY_OP(add, AddOp)

	BINARY_OP(matmul, MatmulOp)

	VariableInterfacePtr pow(VariableInterfacePtr input, float exp)
	{
		DimVector result_dim;
        VariableInterfacePtr result = input->new_empty(result_dim, true);
		std::shared_ptr<PowOp> op(new PowOp);
		op->set_exp(exp);
		result->data_(op->forward({ input->data() }));
		if (input->requires_grad())
		{
			result->grad_(result->data()->create_grad());
			result->grad()->fill_(0);
			result->requires_grad_(true);
			result->grad_fn_(op);
			result->inputs_({ input });
		}
		else
		{
			result->requires_grad_(false);
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

	UNARY_OP(sin, SinOp)

	BINARY_OP(sub, SubOp)

	VariableInterfacePtr transpose(VariableInterfacePtr input, idx_type dim0, idx_type dim1)
	{
		DimVector result_dim;

        VariableInterfacePtr result = input->new_empty(result_dim, true);
		std::shared_ptr<TransposeOp> op(new TransposeOp);
		op->set_dim(dim0, dim1);

		std::vector<VariableInterfacePtr> result_inputs{ input };
		result->data_(op->forward({ input->data() }));

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