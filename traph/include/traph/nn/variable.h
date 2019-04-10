#ifndef TRAPH_NN_VARIABLE_H_
#define TRAPH_NN_VARIABLE_H_

#include <memory>
#include <initializer_list>

#include <traph/core/index.h>
#include <traph/core/tensor.h>
#include <traph/tensor/tensor.h>

namespace traph
{
    template<class T>
    class Variable
    {
    private:
        std::unique_ptr<TensorBase<T>> _data;
        std::unique_ptr<TensorBase<T>> _grad;
    public:
        Variable()
        {

        }

        Variable(const DimVector& dim)
            :_data(new Tensor<T>(dim)), _grad(new Tensor<T>(dim))
        {
        }

        Variable(std::initializer_list<idx_type> l)
            :_data(new Tensor<T>()), _grad(new Tensor<T>())
        {
            DimVector dim;
            for (auto i : l)
                dim.push_back(i);
            
            _data.resize_(dim);
            _grad.resize_(dim);
        }

        ~Variable()
        {

        }
    };
}


#endif