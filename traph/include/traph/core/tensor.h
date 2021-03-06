#ifndef TRAPH_CORE_TENSOR_H_
#define TRAPH_CORE_TENSOR_H_

#include <algorithm>
#include <functional>
#include <memory>

#include <traph/core/type.h>
#include <traph/core/index.h>
#include <traph/core/slice.h>

#include <traph/core/tensor_storage.h>

namespace traph
{
	template<class T>
	class TensorBase;

    class TensorInterface
    {
    public:
        using self_type = TensorInterface;

        using raw_pointer = self_type*;
        using raw_const_pointer = const self_type*;
        using shared_pointer = std::shared_ptr<self_type>;
        using reference = self_type&;
        using const_reference = const self_type&;

    public:
        virtual void add_(shared_pointer other) = 0;
        virtual shared_pointer clone() const = 0;
        virtual void cos_() = 0;
        virtual std::shared_ptr<TensorBase<f32>> create_grad() = 0;
        virtual device_id device() = 0;
        virtual DataType dtype() const = 0;
        virtual bool equal(std::shared_ptr<TensorInterface> other) const = 0;
        virtual std::shared_ptr<TensorInterface> inverse() const = 0;
        virtual std::shared_ptr<TensorInterface> matmul(std::shared_ptr<TensorInterface> mat) const = 0;
        virtual std::shared_ptr<TensorInterface> mean() const = 0;
        virtual void mul_(std::shared_ptr<TensorInterface> other) = 0;
        virtual idx_type ndimension() const = 0;
        virtual void neg_() = 0;
        virtual idx_type offset() const = 0;
        virtual std::shared_ptr<TensorInterface> permute(const DimVector& dims) const = 0;
        virtual PlatformType platform() const = 0;
        virtual void pow_(f32 exp) = 0;
        virtual void reshape_(const DimVector& dims) = 0;
        virtual void resize_(const DimVector& dims) = 0;
        virtual std::shared_ptr<TensorInterface> select(const SliceVector& slice) const = 0;
        virtual void sin_() = 0;
		virtual DimVector size() const = 0;
		virtual idx_type size(idx_type i) const = 0;
		virtual DimVector stride() const = 0;
		virtual idx_type stride(idx_type i) const = 0;
        virtual void sub_(std::shared_ptr<TensorInterface> other) = 0;
        virtual shared_pointer sum() const = 0;
        virtual std::string to_string() const = 0;
        virtual void transpose_(idx_type dim0, idx_type dim1) = 0;
        virtual std::shared_ptr<TensorInterface> transpose(idx_type dim0, idx_type dim1) = 0;
    };

    using TensorInterfacePtr = std::shared_ptr<TensorInterface>;
    using TensorInterfaceRef = TensorInterface&;
    using TensorInterfaceConstRef = const TensorInterface&;

    template<class T>
    class TensorBase: public TensorInterface
    {
    public:
        using value_type = T;
        using self_type = TensorBase<T>;
        using base_type = TensorInterface;

        using raw_pointer = self_type*;
        using raw_const_pointer = const self_type*;
        using shared_pointer = std::shared_ptr<self_type>;
        using reference = self_type&;
        using const_reference = const self_type&;
        
    public:
        virtual void add_(TensorInterfacePtr other) = 0;
        virtual void apply_(std::function<T(T)> f) = 0;
        virtual TensorInterfacePtr clone() const = 0;
        virtual void cos_() = 0;
        virtual std::shared_ptr<TensorBase<f32>> create_grad() = 0;
        virtual T* data_ptr() = 0;
        virtual const T* data_ptr() const = 0;
        virtual device_id device() = 0;
        virtual DataType dtype() const = 0;
        virtual bool equal(std::shared_ptr<TensorInterface> other) const = 0;
        virtual void fill_(T value) = 0;
        virtual std::shared_ptr<TensorInterface> inverse() const = 0;
        virtual T item() const = 0;
        virtual std::shared_ptr<TensorInterface> matmul(std::shared_ptr<TensorInterface> mat) const = 0;
        virtual TensorInterfacePtr mean() const = 0;
        virtual void mul_(T value) = 0;
        virtual void mul_(std::shared_ptr<TensorInterface> other) = 0;
        virtual idx_type ndimension() const = 0;
        virtual void neg_() = 0;
        virtual idx_type offset() const = 0;
        virtual std::shared_ptr<TensorInterface> permute(const DimVector& dims) const = 0;
        virtual PlatformType platform() const = 0;
        virtual void pow_(f32 exp) = 0;
        virtual T reduce(std::function<T(T,T)> f) const = 0;
        virtual TensorInterfacePtr reduce_dim(idx_type dim, std::function<T(T,T)> f) const = 0;
        virtual void reshape_(const DimVector& dims) = 0;
        virtual void resize_(const DimVector& dims) = 0;
        virtual std::shared_ptr<TensorInterface> select(const SliceVector& slice) const = 0;
        virtual void sin_() = 0;
		virtual DimVector size() const = 0;
		virtual idx_type size(idx_type i) const = 0;
        virtual std::shared_ptr<StorageBase<T>> storage() const = 0;
		virtual DimVector stride() const = 0;
		virtual idx_type stride(idx_type i) const = 0;
        virtual void sub_(std::shared_ptr<TensorInterface> other) = 0;
        virtual TensorInterfacePtr sum() const = 0;
        virtual std::string to_string() const = 0;
        virtual void transpose_(idx_type dim0, idx_type dim1) = 0;
        virtual std::shared_ptr<TensorInterface> transpose(idx_type dim0, idx_type dim1) = 0;
    };

    using DoubleTensorBase = TensorBase<f64>;
    using FloatTensorBase = TensorBase<f32>;
    using LongTensorBase = TensorBase<i64>;
    using IntTensorBase = TensorBase<i32>;
    using ShortTensorBase = TensorBase<i16>;
    using CharTensorBase = TensorBase<i8>;
    using ByteTensorBase = TensorBase<u8>;

    template<class T>
    using TensorBasePtr = std::shared_ptr<TensorBase<T>>;
    template<class T>
    using TensorBaseRef = TensorBase<T>&;
    template<class T>
    using TensorBaseConstRef = const TensorBase<T>&;

    bool broadcastable(const DimVector &lhs, const DimVector & rhs);

	DimVector broadcast_shape(const DimVector &lhs, const DimVector & rhs);

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

    template<class T>
    std::shared_ptr<TensorBase<T>> sort_strides(std::shared_ptr<TensorBase<T>> t)
    {
        DimVector indices(t->ndimension());
        for (idx_type i = 0; i < t->ndimension(); i++)
            indices[i] = i;

        std::sort(&indices[0], &indices[indices.size() - 1], std::greater<idx_type>());
        std::shared_ptr<TensorInterface> ret = t->permute(indices);
        return ret;
    }
}

#endif