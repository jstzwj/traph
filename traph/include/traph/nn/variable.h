#ifndef TRAPH_NN_VARIABLE_H_
#define TRAPH_NN_VARIABLE_H_

#include <memory>
#include <initializer_list>

#include <traph/core/index.h>
#include <traph/core/tensor.h>
#include <traph/core/variable.h>
#include <traph/tensor/tensor.h>

namespace traph
{
    template<class T>
    class Variable: public VariableBase<T>
    {
    private:
        std::unique_ptr<TensorBase<T>> _data;
        std::unique_ptr<TensorBase<T>> _grad;
        bool _requires_grad;
    public:
        Variable()
        {

        }

        Variable(const DimVector& dim)
            :_data(new Tensor<T>(dim)), _grad(new Tensor<T>(dim)), _requires_grad(false)
        {
        }

        Variable(std::initializer_list<idx_type> l)
            :_data(new Tensor<T>()), _grad(new Tensor<T>()), _requires_grad(false)
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

        virtual void reshape(const DimVector& dims) override
        {

        }

        virtual void resize(const DimVector& dims) override
        {

        }

		virtual idx_type offset() const override
        {

        }

		virtual layout_type layout() const override
        {

        }

		virtual DimVector size() const override
        {

        }

		virtual const T* data() const override
        {

        }
		virtual T* data() override
        {

        }

		virtual DimVector strides() const override
        {

        }
    };
}


#endif