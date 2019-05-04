#ifndef TRAPH_SPARSE_TENSOR_TENSOR_H_
#define TRAPH_SPARSE_TENSOR_TENSOR_H_

#include <initializer_list>
#include <cmath>
#include <cfenv>
#include <memory>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <vector>
#include <utility>


#include<traph/core/type.h>
#include<traph/core/index.h>
#include<traph/core/tensor.h>

namespace traph
{
    class SparseIndex
    {

    };

    // ndarray
    template<typename T>
    class SparseTensor: public TensorBase<T>
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
        std::vector<std::pair<SparseIndex, T>> _rep;
        DimVector _dimensions;

    public:
        SparseTensor();
        explicit SparseTensor(const DimVector& dimensions);
        explicit SparseTensor(const DimVector& dimensions, layout_type order);
        explicit SparseTensor(const DimVector& dimensions, const DimVector& strides);
        explicit SparseTensor(const DimVector& dimensions, const DimVector& strides, layout_type order);
        SparseTensor(const T& t);

        SparseTensor(const SparseTensor& other) = delete;
        SparseTensor(SparseTensor&& other) = delete;
        SparseTensor& operator= (const SparseTensor& other) = delete;
        SparseTensor& operator= (SparseTensor&& other) = delete;

        virtual void add_(TensorInterfacePtr other) override;
        virtual void apply_(std::function<T(T)> f) override;
        virtual TensorInterfacePtr clone() const override;
        virtual void cos_() override;
        virtual std::shared_ptr<TensorBase<f32>> create_grad() override;
        virtual T* data_ptr() override;
        virtual const T* data_ptr() const override;
        virtual device_id device() override;
        virtual DataType dtype() const override;
        virtual void fill_(T value) override;
        virtual std::shared_ptr<TensorInterface> inverse() const override;
        virtual T item() const override;
        virtual std::shared_ptr<TensorInterface> matmul(std::shared_ptr<TensorInterface> mat) const override;
		virtual void mul_(T value) override;
        virtual void neg_() override;
        virtual idx_type offset() const override;
		virtual layout_type order() const override;
        virtual PlatformType platform() override;
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
	using SparseTensorPtr = std::shared_ptr<SparseTensor<T>>;
	template<typename T>
	using SparseTensorRef = SparseTensor<T> &;
	template<typename T>
	using SparseTensorConstRef = const SparseTensor<T>&;
}

#endif // !TRAPH_SPARSE_TENSOR_TENSOR_H_