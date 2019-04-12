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
#include <traph/core/operation.h>
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
    private:
        std::shared_ptr<TensorBase<T>> _data;
        std::shared_ptr<TensorBase<f32>> _grad;
        bool _requires_grad;
        bool _leaf;
        std::shared_ptr<OpBase> _grad_fn;
        std::vector<VariableInterfacePtr> _inputs;
        // std::vector<std::weak_ptr<VariableInterface>> _outputs;
    public:
        Variable();
        Variable(std::shared_ptr<TensorBase<T>> data);
        Variable(const DimVector& dim);
        Variable(const DimVector& dim, bool is_leaf);
        Variable(std::initializer_list<idx_type> l);

		Variable(const Variable& other) = delete;
		Variable(Variable&& other) = delete;
        Variable& operator= (const Variable& other) = delete;
        Variable& operator= (Variable&& other) = delete;

        ~Variable();

		template<class T>
		friend std::shared_ptr<Variable<T>> sum(std::shared_ptr<Variable<T>> input);

        virtual void backward() override;
        virtual device_id device() override;
        virtual void fill_(T value) override;
        virtual TensorBasePtr<f32> grad() override;
        virtual std::shared_ptr<OpBase> grad_fn() override;
        virtual std::vector<VariableInterfacePtr>& inputs() override;
        virtual T item() const override;
        virtual idx_type offset() const override;
		virtual layout_type order() const override;
        virtual platform_type platform() override;
        virtual void requires_grad_(bool requires_grad) override;
        virtual void reshape_(const DimVector& dims) override;
        virtual void resize_(const DimVector& dims) override;
		virtual DimVector size() const override;
        virtual StorageBase<T>& storage() const override;
		virtual DimVector stride() const override;
    };

    template<class T>
    using VariablePtr = std::shared_ptr<Variable<T>>;
    template<class T>
    using VariableRef = Variable<T>&;
    template<class T>
    using VariableConstRef = const Variable<T>&;

    // only support these type template, so methods defined in source is ok...
    template class Variable<u8>;
    template class Variable<i8>;
    template class Variable<i16>;
    template class Variable<i32>;
    template class Variable<i64>;
    template class Variable<f32>;
    template class Variable<f64>;

    using DoubleVariable = Variable<f64>;
    using FloatVariable = Variable<f32>;
    using LongVariable = Variable<i64>;
    using IntVariable = Variable<i32>;
    using ShortVariable = Variable<i16>;
    using CharVariable = Variable<i8>;
    using ByteVariable = Variable<u8>;

	// definition
	template<typename T>
	Variable<T>::Variable()
		:_data(new Tensor<T>), _grad(nullptr),
		_requires_grad(false), _leaf(false),
		_grad_fn(nullptr), _inputs()
	{

	}

	template<typename T>
	Variable<T>::Variable(std::shared_ptr<TensorBase<T>> data)
		:_data(data), _grad(nullptr),
		_requires_grad(false), _leaf(false),
		_grad_fn(nullptr), _inputs()
	{
	}

	template<typename T>
	Variable<T>::Variable(const DimVector& dim)
		:_data(new Tensor<T>(dim)), _grad(nullptr),
		_requires_grad(false), _leaf(false),
		_grad_fn(nullptr), _inputs()
	{
	}

	template<typename T>
	Variable<T>::Variable(const DimVector& dim, bool is_leaf)
		:_data(new Tensor<T>(dim)), _grad(nullptr),
		_requires_grad(false), _leaf(is_leaf),
		_grad_fn(nullptr), _inputs()
	{
		if (is_leaf)
		{
			_requires_grad = true;

			_grad = _data->create_grad();
		}
	}

	template<typename T>
	Variable<T>::Variable(std::initializer_list<idx_type> l)
		:_data(new Tensor<T>()), _grad(nullptr),
		_requires_grad(false), _leaf(false),
		_grad_fn(nullptr), _inputs()
	{
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

		if (_data)
			_data->resize_(dim);
		if (_grad)
			_grad->resize_(dim);
	}

	template<typename T>
	Variable<T>::~Variable()
	{

	}

	template<typename T>
	void Variable<T>::backward()
	{
		_grad->fill_(1);

		std::vector<VariableInterface*> sorted_node = Executor::topology_sort(dynamic_cast<VariableInterface*>(this));
		for (int i = sorted_node.size() - 1; i >= 0; --i)
		{
			VariableInterface* cur_node = sorted_node[i];
			std::vector<TensorBasePtr<f32>> back_grad = cur_node->grad_fn()->backward(cur_node->grad());

			assert(back_grad.size() == _inputs.size());
			for (int i = 0; i < cur_node->inputs().size(); ++i)
			{
				cur_node->inputs()[i]->grad()->add_(back_grad[i]);
			}
		}

	}

	template<typename T>
	device_id Variable<T>::device()
	{
		return _data->device();
	}

	template<typename T>
	void Variable<T>::fill_(T value)
	{
		return _data->fill_(value);
	}

	template<typename T>
	TensorBasePtr<f32> Variable<T>::grad()
	{
		return _grad;
	}

	template<typename T>
	std::shared_ptr<OpBase> Variable<T>::grad_fn()
	{
		return _grad_fn;
	}

	template<typename T>
	std::vector<VariableInterfacePtr>& Variable<T>::inputs()
	{
		return _inputs;
	}

	template<typename T>
	T Variable<T>::item() const
	{
		return _data->item();
	}

	template<typename T>
	idx_type Variable<T>::offset() const
	{
		return _data->offset();
	}

	template<typename T>
	layout_type Variable<T>::order() const
	{
		return _data->order();
	}

	template<typename T>
	platform_type Variable<T>::platform()
	{
		return _data->platform();
	}

	template<typename T>
	void Variable<T>::requires_grad_(bool requires_grad)
	{
		_requires_grad = requires_grad;
		if (requires_grad)
		{
			_grad = _data->create_grad();
			_grad->fill_(0);
		}
		else
		{
			_grad = std::shared_ptr<TensorBase<f32>>(nullptr);
		}
	}

	template<typename T>
	void Variable<T>::reshape_(const DimVector& dims)
	{
		_data->reshape_(dims);
	}

	template<typename T>
	void Variable<T>::resize_(const DimVector& dims)
	{
		_data->resize_(dims);
	}

	template<typename T>
	DimVector Variable<T>::size() const
	{
		return _data->size();
	}

	template<typename T>
	StorageBase<T>& Variable<T>::storage() const
	{
		return _data->storage();
	}

	template<typename T>
	DimVector Variable<T>::stride() const
	{
		return _data->stride();
	}

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