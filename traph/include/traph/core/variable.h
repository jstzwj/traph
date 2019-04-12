#ifndef TRAPH_CORE_VARIABLE_H_
#define TRAPH_CORE_VARIABLE_H_

#include <functional>

#include <traph/core/type.h>
#include <traph/core/tensor.h>

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
    class VariableBase
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
        virtual void apply_(std::function<T(T)> f) = 0;
        virtual void backward() = 0;
        virtual void cos_() = 0;
        virtual device_id device() = 0;
        virtual void fill_(T value) = 0;
        virtual T item() const = 0;
        virtual idx_type offset() const = 0;
		virtual layout_type order() const = 0;
        virtual platform_type platform() = 0;
        virtual T reduce_(std::function<T(T,T)> f) const = 0;
        virtual void requires_grad_(bool requires_grad) = 0;
        virtual void reshape_(const DimVector& dims) = 0;
        virtual void resize_(const DimVector& dims) = 0;
        virtual void sin_() = 0;
		virtual DimVector size() const = 0;
        virtual StorageBase<T>& storage() const = 0;
		virtual DimVector stride() const = 0;
        virtual VariableBasePtr sum() const = 0;
    };
}

#endif