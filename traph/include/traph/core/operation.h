#ifndef TRAPH_CORE_OPERATION_H_
#define TRAPH_CORE_OPERATION_H_

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
			TensorBasePtr<f32> result = std::dynamic_pointer_cast<TensorBase<f32>>(output_grad->clone());
			result->cos_();
			return { result };
		}
	};
}

#endif