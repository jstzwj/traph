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
#include <traph/core/utils.h>
#include <traph/core/variable.h>
#include <traph/nn/variable.h>
#include <traph/core/tensor.h>
#include <traph/tensor/tensor.h>
#include <traph/nn/graph.h>

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
    };

    template<class T>
    class OpInterface: public OpBase
    {
    public:
        virtual TensorBasePtr<T> forward(std::vector<TensorBasePtr<T>> inputs) = 0;
        virtual std::vector<TensorBasePtr<T>> backward(TensorBasePtr<T> output_grad) = 0;
    };

    template<class T>
    class SumOp: public OpInterface<T>
    {
    public:
        virtual TensorBasePtr<T> forward(std::vector<TensorBasePtr<T>> inputs) override
        {
            assert(inputs.size() == 1);
            
            TensorBasePtr<T> input = inputs[0];
            TensorBasePtr<T> result = input->sum();

			return result;
        }

        virtual std::vector<TensorBasePtr<T>> backward(TensorBasePtr<T> output_grad) override
        {
            return {output_grad};
        }
    };
}

#endif