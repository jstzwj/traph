#ifndef TRAPH_CORE_VARIABLE_H_
#define TRAPH_CORE_VARIABLE_H_

#include <functional>
#include <vector>

#include <traph/core/type.h>
#include <traph/core/tensor.h>
#include <traph/core/operation.h>

namespace traph
{
    class VariableInterface
    {
    public:
        using VariableInterfacePtr = std::shared_ptr<VariableInterface>;
        using VariableInterfaceRef = VariableInterface&;
        using VariableInterfaceConstRef = const VariableInterface&;

    public:
        virtual void backward() = 0;
        virtual device_id device() = 0;
        virtual TensorBasePtr<f32> grad() = 0;
        virtual std::shared_ptr<OpBase> grad_fn() = 0;
        virtual std::vector<VariableInterfacePtr>& inputs() = 0;
        virtual bool is_leaf() const = 0;
        virtual idx_type offset() const = 0;
		virtual layout_type order() const = 0;
        virtual platform_type platform() = 0;
        virtual void requires_grad_(bool requires_grad) = 0;
        virtual void reshape_(const DimVector& dims) = 0;
        virtual void resize_(const DimVector& dims) = 0;
		virtual DimVector size() const = 0;
		virtual DimVector stride() const = 0;
    };

    using VariableInterfacePtr = std::shared_ptr<VariableInterface>;
    using VariableInterfaceRef = VariableInterface&;
    using VariableInterfaceConstRef = const VariableInterface&;

    template<class T>
    class VariableBase: public VariableInterface
    {
    public:
        using VariableBasePtr = std::shared_ptr<VariableBase<T>>;
        using VariableBaseRef = VariableBase<T>&;
        using VariableBaseConstRef = const VariableBase<T>&;

        using DoubleVariableBase = VariableBase<f64>;
        using FloatVariableBase = VariableBase<f32>;
        using LongVariableBase = VariableBase<i64>;
        using IntVariableBase = VariableBase<i32>;
        using ShortVariableBase = VariableBase<i16>;
        using CharVariableBase = VariableBase<i8>;
        using ByteVariableBase = VariableBase<u8>;
    public:
        virtual void backward() = 0;
        virtual device_id device() = 0;
        virtual void fill_(T value) = 0;
        virtual TensorBasePtr<f32> grad() = 0;
        virtual std::shared_ptr<OpBase> grad_fn() = 0;
        virtual std::vector<VariableInterfacePtr>& inputs() = 0;
        virtual bool is_leaf() const = 0;
        virtual T item() const = 0;
        virtual idx_type offset() const = 0;
		virtual layout_type order() const = 0;
        virtual platform_type platform() = 0;
        virtual void requires_grad_(bool requires_grad) = 0;
        virtual void reshape_(const DimVector& dims) = 0;
        virtual void resize_(const DimVector& dims) = 0;
		virtual DimVector size() const = 0;
        virtual StorageBase<T>& storage() const = 0;
		virtual DimVector stride() const = 0;
    };

	template<typename T>
    using VariableBasePtr = std::shared_ptr<VariableBase<T>>;
	template<typename T>
    using VariableBaseRef = VariableBase<T>&;
	template<typename T>
    using VariableBaseConstRef = const VariableBase<T>&;
}

#endif