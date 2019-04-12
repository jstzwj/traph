#ifndef TRAPH_CORE_TENSOR_H_
#define TRAPH_CORE_TENSOR_H_

#include <algorithm>
#include <functional>
#include <memory>

#include <traph/core/type.h>
#include <traph/core/index.h>


namespace traph
{
    template<class T>
    class StorageBase
    {
    public:
        virtual size_type element_size() const = 0;
        virtual void fill_(T v) = 0;
        virtual void resize_(idx_type size) = 0;
        virtual idx_type size() const = 0;

    };

    template<class T>
    class ContiguousStorageBase: public StorageBase<T>
    {
    public:
        virtual size_type element_size() const = 0;
        virtual void fill_(T v) = 0;
        virtual void resize_(idx_type size) = 0;
        virtual idx_type size() const = 0;
    };

    class TensorInterface
    {
    public:
        using TensorInterfacePtr = std::shared_ptr<TensorInterface>;
        using TensorInterfaceRef = TensorInterface&;
        using TensorInterfaceConstRef = const TensorInterface&;

    public:
        virtual device_id device() = 0;
        virtual idx_type offset() const = 0;
		virtual layout_type order() const = 0;
        virtual platform_type platform() = 0;
        virtual void reshape_(const DimVector& dims) = 0;
        virtual void resize_(const DimVector& dims) = 0;
		virtual DimVector size() const = 0;
		virtual DimVector stride() const = 0;
    };

    using TensorInterfacePtr = std::shared_ptr<TensorInterface>;
    using TensorInterfaceRef = TensorInterface&;
    using TensorInterfaceConstRef = const TensorInterface&;

    template<class T>
    class TensorBase: public TensorInterface
    {
    public:
        using TensorBasePtr = std::shared_ptr<TensorBase<T>>;
        using TensorBaseRef = TensorBase<T>&;
        using TensorBaseConstRef = const TensorBase<T>&;

        using DoubleTensorBase = TensorBase<f64>;
        using FloatTensorBase = TensorBase<f32>;
        using LongTensorBase = TensorBase<i64>;
        using IntTensorBase = TensorBase<i32>;
        using ShortTensorBase = TensorBase<i16>;
        using CharTensorBase = TensorBase<i8>;
        using ByteTensorBase = TensorBase<u8>;
    public:
        virtual void apply_(std::function<T(T)> f) = 0;
        virtual void cos_() = 0;
        virtual TensorBasePtr create_grad() = 0;
        virtual device_id device() = 0;
        virtual void fill_(T value) = 0;
        virtual T item() const = 0;
        virtual idx_type offset() const = 0;
		virtual layout_type order() const = 0;
        virtual platform_type platform() = 0;
        virtual T reduce_(std::function<T(T,T)> f) const = 0;
        virtual TensorBasePtr reduce_dim(idx_type dim, std::function<T(T,T)> f) const = 0;
        virtual void reshape_(const DimVector& dims) = 0;
        virtual void resize_(const DimVector& dims) = 0;
        virtual void sin_() = 0;
		virtual DimVector size() const = 0;
        virtual StorageBase<T>& storage() const = 0;
		virtual DimVector stride() const = 0;
        virtual TensorBasePtr sum() const = 0;
    };

    template<class T>
    using TensorBasePtr = std::shared_ptr<TensorBase<T>>;
    template<class T>
    using TensorBaseRef = TensorBase<T>&;
    template<class T>
    using TensorBaseConstRef = const TensorBase<T>&;

    template<class T>
    bool broadcastable(const TensorBase<T> &lhs, const TensorBase<T> & rhs)
    {
        DimVector lhs_dim = lhs.size();
        DimVector rhs_dim = rhs.size();
        if(lhs_dim.size() < 1 || rhs_dim.size() < 1)
            return false;

        idx_type min = std::min(lhs_dim.size(), rhs_dim.size());
        for(idx_type i = 0; i<min;++i)
        {
            if(lhs_dim[i] != rhs_dim[i] &&
                lhs_dim[i] != 1 &&
                rhs_dim[i] != 1)
            {
                return false;
            }
        }

        return true;
    }

    template<class T>
    bool strict_same_shape(const TensorBase<T> &lhs, const TensorBase<T> & rhs)
    {
        DimVector lhs_dim = lhs.size();
        DimVector rhs_dim = rhs.size();
        if(lhs_dim.size() != rhs_dim.size())
            return false;
        for(idx_type i = 0; i < lhs_dim.size(); ++i)
        {
            if(lhs_dim[i] != rhs_dim[i])
                return false;
        }

        return true;
    }
}

#endif