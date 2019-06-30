#ifndef TRAPH_TENSOR_TENSOR_H_
#define TRAPH_TENSOR_TENSOR_H_

#include <initializer_list>
#include <cmath>
#include <cfenv>
#include <memory>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <string>


#include<traph/core/type.h>
#include<traph/core/index.h>
#include<traph/core/utils.h>
#include<traph/core/tensor.h>

#include<traph/tensor/tensor_storage.h>
#include<traph/tensor/arithmetic.h>

namespace traph
{
    // ndarray
    template<typename T>
    class Tensor: public TensorBase<T>
    {
    public:
        using value_type = T;
        using self_type = Tensor<T>;
        using base_type = TensorBase<T>;
        using storage_type = TensorStorage<value_type>;

        using raw_pointer = self_type*;
        using raw_const_pointer = const self_type*;
        using shared_pointer = std::shared_ptr<self_type>;
        using reference = self_type&;
        using const_reference = const self_type&;
    private:
        std::shared_ptr<TensorStorage<T>> _rep;
        DimVector _dimensions;
        idx_type _offset;
		DimVector _strides;

    private:
        void auto_strides();
        void reduce_impl(T& result, idx_type dim, idx_type idx, std::function<T(T,T)> f) const;
		void Tensor<T>::reduce_dim_impl(Tensor<T>& result, idx_type dim, idx_type reduce_dim,
			idx_type this_idx, idx_type result_idx,
			std::function<T(T, T)> f) const;
        T reduce_dim_kernel(idx_type begin, idx_type step_len, idx_type step_num, std::function<T(T,T)> f) const;
    public:
        Tensor();
        explicit Tensor(const DimVector& dimensions);
        explicit Tensor(const DimVector& dimensions, const DimVector& strides);
        Tensor(const T& t);

        Tensor(const Tensor& other) = delete;
        Tensor(Tensor&& other) = delete;
        Tensor& operator= (const Tensor& other) = delete;
        Tensor& operator= (Tensor&& other) = delete;

        virtual void add_(TensorInterfacePtr other) override;
        virtual void apply_(std::function<T(T)> f) override;
        virtual TensorInterfacePtr clone() const override;
        virtual void cos_() override;
        virtual std::shared_ptr<TensorBase<f32>> create_grad() override;
        virtual T* data_ptr() override;
        virtual const T* data_ptr() const override;
        virtual device_id device() override;
        virtual DataType dtype() const override;
        virtual bool equal(std::shared_ptr<TensorInterface> other) const override;
        virtual void fill_(T value) override;
        virtual std::shared_ptr<TensorInterface> inverse() const override;
        virtual T item() const override;
        virtual std::shared_ptr<TensorInterface> matmul(std::shared_ptr<TensorInterface> mat) const override;
		virtual TensorInterfacePtr mean() const override;
        virtual void mul_(T value) override;
        virtual void mul_(std::shared_ptr<TensorInterface> other) override;
        virtual idx_type ndimension() const override;
        virtual void neg_() override;
        virtual idx_type offset() const override;
        virtual std::shared_ptr<TensorInterface> permute(const DimVector& dims) const override;
        virtual PlatformType platform() const override;
        virtual void pow_(f32 exp) override;
        virtual T reduce(std::function<T(T,T)> f) const override;
        virtual TensorInterfacePtr reduce_dim(idx_type dim, std::function<T(T,T)> f) const override;
        virtual void reshape_(const DimVector& dims) override;
        virtual void resize_(const DimVector& dims) override;
        virtual std::shared_ptr<TensorInterface> select(const SliceVector& slice) const override;
        virtual void sin_() override;
		virtual DimVector size() const override;
		virtual idx_type size(idx_type i) const override;
        virtual std::shared_ptr<StorageBase<T>> storage() const override;
		virtual DimVector stride() const override;
		virtual idx_type stride(idx_type i) const override;
        virtual void sub_(std::shared_ptr<TensorInterface> other) override;
        virtual TensorInterfacePtr sum() const override;
        virtual std::string to_string() const override;
        virtual void transpose_(idx_type dim0, idx_type dim1) override;
        virtual std::shared_ptr<TensorInterface> transpose(idx_type dim0, idx_type dim1) override;
    };

	template<typename T>
	using TensorPtr = std::shared_ptr<Tensor<T>>;
	template<typename T>
	using TensorRef = Tensor<T> &;
	template<typename T>
	using TensorConstRef = const Tensor<T>&;


    template class Tensor<u8>;
    template class Tensor<i8>;
    template class Tensor<i16>;
    template class Tensor<i32>;
    template class Tensor<i64>;
    template class Tensor<f32>;
    template class Tensor<f64>;

    using ByteTensor = Tensor<u8>;
    using CharTensor = Tensor<i8>;
    using ShortTensor = Tensor<i16>;
    using IntTensor = Tensor<i32>;
    using LongTensor = Tensor<i64>;
    using FloatTensor = Tensor<f32>;
    using DoubleTensor = Tensor<f64>;

    // TODO: macros
    // apply apply2 reduce...

#define TENSOR_APPLY_CONTIGUOUS(TYPE, TENSOR, CODE) \
    {                                               \
        TYPE *element_ptr = TENSOR->data_ptr();     \
        idx_type len = TENSOR->storage()->size();   \
        TYPE *end_ptr = element_ptr + len;          \
        while (element_ptr < end_ptr)               \
        {                                           \
            CODE;                                   \
            element_ptr++;                          \
        }                                           \
    }

#define MEMORY_APPLY_CONTIGUOUS(TYPE, PTR, LEN, CODE) \
    {                                                 \
        TYPE *end_ptr = PTR + LEN;                    \
        while (PTR < end_ptr)                         \
        {                                             \
            CODE;                                     \
            PTR++;                                    \
        }                                             \
    }

#define TENSOR_APPLY(TYPE, TENSOR, CODE)                                                                                        \
    {                                                                                                                           \
        idx_type dim_size = TENSOR->ndimension();                                                                               \
        std::function<void(idx_type, TYPE *)> apply_impl =                                                                      \
            [&](idx_type dim, TYPE *element_ptr) {                                                                              \
                idx_type contiguous_dim = dim;                                                                                  \
                while (contiguous_dim < dim_size - 1 &&                                                                         \
                       TENSOR->stride(contiguous_dim + 1) * TENSOR->size(contiguous_dim + 1) == TENSOR->stride(contiguous_dim)) \
                {                                                                                                               \
                    contiguous_dim++;                                                                                           \
                }                                                                                                               \
                                                                                                                                \
                int step_num = 1;                                                                                               \
                for (idx_type i = contiguous_dim; i >= dim; --i)                                                                \
                    step_num *= TENSOR->size(i);                                                                                \
                                                                                                                                \
                if (contiguous_dim == dim_size - 1)                                                                             \
                {                                                                                                               \
                    MEMORY_APPLY_CONTIGUOUS(TYPE, element_ptr, step_num, CODE);                                                 \
                }                                                                                                               \
                else                                                                                                            \
                {                                                                                                               \
                    for (idx_type i = 0; i < step_num; ++i)                                                                     \
                    {                                                                                                           \
                        apply_impl(contiguous_dim + 1, element_ptr);                                                            \
                        element_ptr += TENSOR->stride(contiguous_dim);                                                          \
                    }                                                                                                           \
                }                                                                                                               \
            };                                                                                                                  \
                                                                                                                                \
        apply_impl(0, TENSOR->data_ptr() + TENSOR->offset());                                                                   \
    }
}

#endif // !TRAPH_TENSOR