#ifndef TRAPH_NN_OPERATION_H_
#define TRAPH_NN_OPERATION_H_

#include <utility>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <cassert>

#include <traph/core/type.h>
#include <traph/core/index.h>
#include <traph/core/tensor.h>
#include <traph/tensor/tensor.h>

namespace traph
{
    class OpContext
    {
    private:
        std::vector<TensorInterfacePtr> _saved_tensors;
    public:
        void save(TensorInterfacePtr tensor)
        {
            _saved_tensors.push_back(tensor);
        }

        std::vector<TensorInterfacePtr> get_saved_tensors() const
        {
            return _saved_tensors;
        }
    };

    class OpBase
    {
    public:
        OpContext context;
        
        virtual TensorInterfacePtr forward(std::vector<TensorInterfacePtr> inputs) = 0;
        virtual std::vector<TensorBasePtr<f32>> backward(TensorBasePtr<f32> output_grad) = 0;
    };

	class AddOp : public OpBase
	{
	public:
		virtual TensorInterfacePtr forward(std::vector<TensorInterfacePtr> inputs) override
		{
			assert(inputs.size() == 2);

			TensorInterfacePtr left_input = inputs[0];
			TensorInterfacePtr right_input = inputs[1];
			TensorInterfacePtr result = left_input->clone();
            result->add_(right_input);

			return result;
		}

		virtual std::vector<TensorBasePtr<f32>> backward(TensorBasePtr<f32> output_grad) override
		{
			return { output_grad, output_grad };
		}
	};

	class MatmulOp : public OpBase
	{
	public:
		virtual TensorInterfacePtr forward(std::vector<TensorInterfacePtr> inputs) override
		{
			assert(inputs.size() == 2);

			TensorInterfacePtr left_input = inputs[0];
			TensorInterfacePtr right_input = inputs[1];
			TensorInterfacePtr result = left_input->matmul(right_input);

			context.save(left_input);
			context.save(right_input);

			return result;
		}

		virtual std::vector<TensorBasePtr<f32>> backward(TensorBasePtr<f32> output_grad) override
		{
			auto saved_tensors = context.get_saved_tensors();
			assert(saved_tensors.size() == 2);
			std::shared_ptr<TensorBase<f32>> left_out = std::dynamic_pointer_cast<TensorBase<f32>>(output_grad->matmul(saved_tensors[1]->transpose(0, 1)));
			std::shared_ptr<TensorBase<f32>> right_out = std::dynamic_pointer_cast<TensorBase<f32>>(saved_tensors[0]->transpose(0, 1)->matmul(output_grad));
			return { left_out, right_out };
		}
	};

	class MeanOp : public OpBase
	{
	public:
		virtual TensorInterfacePtr forward(std::vector<TensorInterfacePtr> inputs) override
		{
			assert(inputs.size() == 1);

			TensorInterfacePtr input = inputs[0];
			TensorInterfacePtr result = input->mean();

			context.save(input);

			return result;
		}

		virtual std::vector<TensorBasePtr<f32>> backward(TensorBasePtr<f32> output_grad) override
		{
			auto saved_tensors = context.get_saved_tensors();
			assert(saved_tensors.size() == 1);

			auto flat_size = saved_tensors[0]->size().flat_size();
			auto result = std::dynamic_pointer_cast<TensorBase<f32>>(output_grad->clone());
			result->mul_(1.f/flat_size);
			return { result };
		}
	};

	class PowOp: public OpBase
	{
	private:
		float _exp;
	public:
		void set_exp(float exp)
		{
			_exp = exp;
		}

		virtual TensorInterfacePtr forward(std::vector<TensorInterfacePtr> inputs) override
		{
			assert(inputs.size() == 1);

			TensorInterfacePtr input = inputs[0];
			auto output = input->clone();
			output->pow_(_exp);

			context.save(input);
			
			return output;
		}

		virtual std::vector<TensorBasePtr<f32>> backward(TensorBasePtr<f32> output_grad) override
		{
			auto saved_tensors = context.get_saved_tensors();
			assert(saved_tensors.size() == 1);
			auto cloned_x = std::dynamic_pointer_cast<TensorBase<f32>>(saved_tensors[0]->clone());
			
			//FIXME x^n = n*x^(n-1)
			cloned_x->mul_(_exp);
			cloned_x->mul_(output_grad);
			
			return { cloned_x };
		}
	};

	class SelectOp : public OpBase
	{
	public:
		SliceVector slice;
		void set_slice(const SliceVector& s)
		{
			slice = s;
		}
		virtual TensorInterfacePtr forward(std::vector<TensorInterfacePtr> inputs) override
		{
			assert(inputs.size() == 1);

			TensorInterfacePtr input = inputs[0];
			auto grad = input->create_grad();
			grad->fill_(0);
			auto zero_grad = std::dynamic_pointer_cast<TensorInterface>(grad);

			context.save(zero_grad);
			
			return input->select(slice);
		}

		virtual std::vector<TensorBasePtr<f32>> backward(TensorBasePtr<f32> output_grad) override
		{
			auto saved_tensors = context.get_saved_tensors();
			assert(saved_tensors.size() == 1);
			auto grad = std::dynamic_pointer_cast<TensorBase<f32>>(saved_tensors[0]);
			auto selected_grad = std::dynamic_pointer_cast<TensorBase<f32>>(grad->select(slice));
			selected_grad->add_(output_grad);
			return { grad };
		}
	};

	class SinOp : public OpBase
	{
	public:
		virtual TensorInterfacePtr forward(std::vector<TensorInterfacePtr> inputs) override
		{
			assert(inputs.size() == 1);

			TensorInterfacePtr input = inputs[0];
			TensorInterfacePtr result = input->clone();
			result->sin_();

			return result;
		}

		virtual std::vector<TensorBasePtr<f32>> backward(TensorBasePtr<f32> output_grad) override
		{
			// fixme: bug
			TensorBasePtr<f32> result = std::dynamic_pointer_cast<TensorBase<f32>>(output_grad->clone());
			result->cos_();
			return { result };
		}
	};

	class SubOp : public OpBase
	{
	public:
		virtual TensorInterfacePtr forward(std::vector<TensorInterfacePtr> inputs) override
		{
			assert(inputs.size() == 2);

			TensorInterfacePtr left_input = inputs[0];
			TensorInterfacePtr right_input = inputs[1];
			TensorInterfacePtr result = left_input->clone();
            result->sub_(right_input);

			return result;
		}

		virtual std::vector<TensorBasePtr<f32>> backward(TensorBasePtr<f32> output_grad) override
		{
			auto left = output_grad;
			auto right = output_grad->clone();
			right->neg_();
			return { output_grad, std::dynamic_pointer_cast<TensorBase<f32>>(right) };
		}
	};

	class SumOp: public OpBase
    {
    public:
        virtual TensorInterfacePtr forward(std::vector<TensorInterfacePtr> inputs) override
        {
            assert(inputs.size() == 1);
            
			TensorInterfacePtr input = inputs[0];
			TensorInterfacePtr result = input->sum();

			return result;
        }

        virtual std::vector<TensorBasePtr<f32>> backward(TensorBasePtr<f32> output_grad) override
        {
            return {output_grad};
        }
    };

	class TransposeOp : public OpBase
	{
	private:
		idx_type dim0, dim1;
	public:
		void set_dim(idx_type d0, idx_type d1)
		{
			dim0 = d0;
			dim1 = d1;
		}

		virtual TensorInterfacePtr forward(std::vector<TensorInterfacePtr> inputs) override
		{
			assert(inputs.size() == 1);

			TensorInterfacePtr input = inputs[0];
			TensorInterfacePtr result = input->transpose(dim0, dim1);

			return result;
		}

		virtual std::vector<TensorBasePtr<f32>> backward(TensorBasePtr<f32> output_grad) override
		{
			TensorBasePtr<f32> result = std::dynamic_pointer_cast<TensorBase<f32>>(output_grad->transpose(dim0, dim1));
			return { result };
		}
	};
}

#endif