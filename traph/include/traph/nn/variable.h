#ifndef TRAPH_NN_VARIABLE_H_
#define TRAPH_NN_VARIABLE_H_

#include <memory>
#include <functional>
#include <initializer_list>
#include <vector>
#include <list>
#include <cassert>

#include <traph/core/index.h>
#include <traph/core/tensor.h>
#include <traph/core/variable.h>
#include <traph/tensor/tensor.h>
#include <traph/nn/operation.h>
#include <traph/nn/executor.h>

namespace traph
{
    template<class T>
    class Variable: public VariableBase<T>
    {
    public:
        using VariablePtr = std::shared_ptr<Variable<T>>;
        using VariableRef = Variable<T>&;
        using VariableConstRef = const Variable<T>&;

        using DoubleVariable = Variable<f64>;
        using FloatVariable = Variable<f32>;
        using LongVariable = Variable<i64>;
        using IntVariable = Variable<i32>;
        using ShortVariable = Variable<i16>;
        using CharVariable = Variable<i8>;
        using ByteVariable = Variable<u8>;
    private:
        std::shared_ptr<TensorBase<T>> _data;
        std::shared_ptr<TensorBase<f32>> _grad;
        bool _requires_grad;
        bool _leaf;
        std::shared_ptr<OpInterface<T>> _grad_fn;
        std::vector<VariableInterfacePtr> _inputs;
        std::vector<std::weak_ptr<VariableInterfacePtr>> _outputs;
    public:
        Variable()
            :_data(new Tensor<T>), _grad(nullptr),
            _requires_grad(false), _leaf(false),
            _grad_fn(nullptr), _inputs(), _outputs()
        {

        }

        Variable(std::shared_ptr<TensorBase<T>> data)
            :_data(data), _grad(nullptr),
            _requires_grad(false), _leaf(false),
            _grad_fn(nullptr), _inputs(), _outputs()
        {
        }

        Variable(const DimVector& dim)
            :_data(new Tensor<T>(dim)), _grad(nullptr),
            _requires_grad(false), _leaf(false),
            _grad_fn(nullptr), _inputs(), _outputs()
        {
        }

        Variable(const DimVector& dim, bool is_leaf)
            :_data(new Tensor<T>(dim)), _grad(nullptr),
            _requires_grad(false), _leaf(is_leaf),
            _grad_fn(nullptr), _inputs(), _outputs()
        {
            if(is_leaf)
            {
                _requires_grad = true;

                _grad = _data->create_grad();
            }
        }

        Variable(std::initializer_list<idx_type> l)
            :_data(new Tensor<T>()), _grad(nullptr),
            _requires_grad(false), _leaf(false),
            _grad_fn(nullptr), _inputs(), _outputs()
        {
            DimVector dim;
            for (auto i : l)
                dim.push_back(i);
            
			if (_data)
				_data->resize_(dim);
			if(_grad)
				_grad->resize_(dim);
        }

		Variable(const Variable& other) = delete;

		Variable(Variable&& other) = delete;

        Variable& operator= (const Variable& other) = delete;

        Variable& operator= (Variable&& other) = delete;

        ~Variable()
        {

        }

		template<class T>
		friend std::shared_ptr<Variable<T>> sum(std::shared_ptr<Variable<T>> input);

        virtual void backward() override
        {
            _grad->fill_(1);
            
            std::vector<VariableInterface*> sorted_node = Executor::topology_sort(this);
            for(int i = sorted_node.size() - 1; i >=0; --i)
            {
                VariableInterface* cur_node = sorted_node[i];
                std::vector<TensorBasePtr<T>> back_grad = cur_node->_grad_fn->backward(cur_node->grad());

                assert(back_grad.size() == _inputs.size());
                for(int i = 0; i < cur_node->inputs().size(); ++i)
                {
                    cur_node->inputs()[i]->grad().add_(back_grad[i]);
                }
            }

        }
        virtual device_id device() override
        {
            return _data->device();
        }
        virtual void fill_(T value) override
        {
            return _data->fill_(value);
        }
        virtual TensorBasePtr<f32> grad() override
        {
            return _grad;
        }
        virtual std::vector<VariableInterfacePtr>& inputs() override
        {
            return _inputs;
        }
        virtual T item() const override
        {
            return _data->item();
        }
        virtual idx_type offset() const override
        {
            return _data->offset();
        }
		virtual layout_type order() const override
        {
            return _data->order();
        }
        virtual std::vector<std::weak_ptr<VariableInterface>>& outputs() override
        {
            return _outputs;
        }
        virtual platform_type platform() override
        {
            return _data->platform();
        }
        virtual void requires_grad_(bool requires_grad) override
        {
            _requires_grad = requires_grad;
            if(requires_grad)
            {
                _grad = _data->create_grad();
                _grad->fill_(0);
            }
            else
            {
                _grad = std::shared_ptr<TensorBase<T>>(nullptr);
            }
        }
        virtual void reshape_(const DimVector& dims) override
        {
            _data->reshape_(dims);
        }
        virtual void resize_(const DimVector& dims) override
        {
            _data->resize_(dims);
        }
		virtual DimVector size() const override
        {
            return _data->size();
        }
        virtual StorageBase<T>& storage() const override
        {
            return _data->storage();
        }
		virtual DimVector stride() const override
        {
            return _data->stride();
        }
        
    };

    template<class T>
    using VariablePtr = std::shared_ptr<Variable<T>>;

    template<class T>
    using VariableRef = Variable<T>&;

    template<class T>
    using VariableConstRef = const Variable<T>&;

    // variable constructor
    template<class T>
    VariablePtr<T> zeros(std::initializer_list<idx_type> l, bool requires_grad = false)
    {
        DimVector dim;
		for (auto i : l)
			dim.push_back(i);

        std::shared_ptr<Variable<T>> result(new Variable<T>(dim, true));
        result->fill_(0);

        return result;
    }

    template<class T>
    VariablePtr<T> ones(std::initializer_list<idx_type> l, bool requires_grad = false)
    {
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

        std::shared_ptr<Variable<T>> result(new Variable<T>(dim, true));
        result->fill_(1);

        return result;
    }
}


#endif