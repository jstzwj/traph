#ifndef TRAPH_TENSOR_BYTE_TENSOR_H_
#define TRAPH_TENSOR_BYTE_TENSOR_H_

#include <utility>
#include <cmath>

#include <traph/tensor/tensor.h>

namespace traph
{
    // ndarray
    template<>
    class Tensor<u8>: public TensorBase<u8>
    {
    public:
        using value_type = u8;
        using self_type = Tensor<u8>;
        using base_type = TensorBase<u8>;
        using storage_type = TensorStorage<value_type>;

        using raw_pointer = self_type*;
        using shared_pointer = std::shared_ptr<self_type>;
        using reference = self_type&;
        using const_reference = const self_type&;
    private:
        std::shared_ptr<storage_type> _rep;
        DimVector _dimensions;
        idx_type _offset;
		DimVector _strides;
        layout_type _order;

    private:
        void auto_strides();

        void apply_impl(idx_type dim, idx_type idx, std::function<value_type(value_type)> f);

        void reduce_impl(value_type& result, idx_type dim, idx_type idx, std::function<value_type(value_type,value_type)> f) const;

        value_type reduce_dim_kernel(idx_type begin, idx_type step_len, idx_type step_num, std::function<value_type(value_type,value_type)> f) const;

        void reduce_dim_impl(reference result, idx_type dim, idx_type reduce_dim,
            idx_type this_idx, idx_type result_idx,
            std::function<value_type(value_type,value_type)> f) const;
    public:
        Tensor();
        explicit Tensor(const DimVector& dimensions);
        explicit Tensor(const DimVector& dimensions, layout_type order);
        explicit Tensor(const DimVector& dimensions, const DimVector& strides);
        explicit Tensor(const DimVector& dimensions, const DimVector& strides, layout_type order);
        Tensor(const value_type& t);

        Tensor(const Tensor& other) = delete;
        Tensor(Tensor&& other) = delete;
        Tensor& operator= (const Tensor& other) = delete;
        Tensor& operator= (Tensor&& other) = delete;

		virtual void add_(TensorInterfacePtr other) override;
		virtual void apply_(std::function<u8(u8)> f) override;
		virtual TensorInterfacePtr clone() const override;
		virtual void cos_() override;
		virtual std::shared_ptr<TensorBase<f32>> create_grad() override;
		virtual u8* data_ptr() override;
		virtual const u8* data_ptr() const override;
		virtual device_id device() override;
		virtual void fill_(u8 value) override;
		virtual std::shared_ptr<TensorInterface> inverse() const override;
		virtual u8 item() const override;
		virtual std::shared_ptr<TensorInterface> matmul(std::shared_ptr<TensorInterface> mat) const override;
		virtual idx_type offset() const override;
		virtual layout_type order() const override;
		virtual platform_type platform() override;
		virtual u8 reduce_(std::function<u8(u8, u8)> f) const override;
		virtual TensorInterfacePtr reduce_dim(idx_type dim, std::function<u8(u8, u8)> f) const override;
		virtual void reshape_(const DimVector& dims) override;
		virtual void resize_(const DimVector& dims) override;
		virtual std::shared_ptr<TensorInterface> select(const SliceVector& slice) const override;
		virtual void sin_() override;
		virtual DimVector size() const override;
		virtual idx_type size(idx_type i) const override;
		virtual std::shared_ptr<StorageBase<u8>> storage() const override;
		virtual DimVector stride() const override;
		virtual idx_type stride(idx_type i) const override;
		virtual TensorInterfacePtr sum() const override;
		virtual std::string to_string() const override;
    };

    using ByteTensor = Tensor<u8>;
}

#endif