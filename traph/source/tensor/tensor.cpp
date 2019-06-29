#include <traph/tensor/tensor.h>


namespace traph
{
    // definition
    // public
    template<typename T>
    Tensor<T>::Tensor()
        :_rep(new TensorStorage<T>),
        _dimensions(), _offset(0), _strides(), _order(layout_type::row_major)
    {
        throw std::runtime_error("No implement");
    }


    template<typename T>
    Tensor<T>::Tensor(const DimVector& dimensions)
        :_rep(new TensorStorage<T>),
        _dimensions(dimensions), _offset(0), _strides(), _order(layout_type::row_major)
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    Tensor<T>::Tensor(const DimVector& dimensions, layout_type order)
        :_rep(new TensorStorage<T>),
        _dimensions(dimensions), _offset(0), _strides(), _order(order)
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    Tensor<T>::Tensor(const DimVector& dimensions, const DimVector& strides)
        :_rep(new TensorStorage<T>),
        _dimensions(dimensions), _offset(0), _strides(strides), _order(layout_type::row_major)
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    Tensor<T>::Tensor(const DimVector& dimensions, const DimVector& strides, layout_type order)
        :_rep(new TensorStorage<T>),
        _dimensions(dimensions), _offset(0), _strides(strides), _order(order)
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    Tensor<T>::Tensor(const T& t)
        :_rep(new TensorStorage<T>),
        _dimensions(), _offset(0), _strides(), _order(layout_type::row_major)
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    void Tensor<T>::add_(TensorInterfacePtr other)
    {
		throw std::runtime_error("No implement");
    }
    template<typename T>
    void Tensor<T>::apply_(std::function<T(T)> f)
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    TensorInterfacePtr Tensor<T>::clone() const
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    void Tensor<T>::cos_()
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    std::shared_ptr<TensorBase<f32>> Tensor<T>::create_grad()
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    T* Tensor<T>::data_ptr()
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    const T* Tensor<T>::data_ptr() const
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    device_id Tensor<T>::device() { throw std::runtime_error("No implement"); }

    template<typename T>
    DataType Tensor<T>::dtype() const
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    bool Tensor<T>::equal(std::shared_ptr<TensorInterface> other) const
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    void Tensor<T>::fill_(T value)
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    std::shared_ptr<TensorInterface> Tensor<T>::inverse() const
    {
        // return std::dynamic_pointer_cast<TensorInterface>(inverse_impl(*this));
		throw std::runtime_error("No implement");
    }

    template<typename T>
    T Tensor<T>::item() const
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    std::shared_ptr<TensorInterface> Tensor<T>::matmul(std::shared_ptr<TensorInterface> mat) const
    {
		throw std::runtime_error("No implement");
    }

    template<typename T>
    TensorInterfacePtr Tensor<T>::mean() const
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    void Tensor<T>::mul_(T value)
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    void Tensor<T>::mul_(std::shared_ptr<TensorInterface> other)
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    idx_type Tensor<T>::ndimension() const
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    void Tensor<T>::neg_()
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    idx_type Tensor<T>::offset() const { throw std::runtime_error("No implement"); }
    template<typename T>
    layout_type Tensor<T>::order() const { throw std::runtime_error("No implement"); }
    template<typename T>
    std::shared_ptr<TensorInterface> Tensor<T>::permute(const DimVector& dims) const
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    PlatformType Tensor<T>::platform() const { throw std::runtime_error("No implement"); }
    template<typename T>
    void Tensor<T>::pow_(f32 exp)
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    T Tensor<T>::reduce(std::function<T(T,T)> f) const
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    TensorInterfacePtr Tensor<T>::reduce_dim(idx_type dim, std::function<T(T,T)> f) const
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    void Tensor<T>::reshape_(const DimVector& dims)
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    void Tensor<T>::resize_(const DimVector& dims)
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    std::shared_ptr<TensorInterface> Tensor<T>::select(const SliceVector& slice) const
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    void Tensor<T>::sin_()
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    DimVector Tensor<T>::size() const { throw std::runtime_error("No implement"); }
	template<typename T>
	idx_type Tensor<T>::size(idx_type i) const
	{ 
		throw std::runtime_error("No implement");
	}
    template<typename T>
	std::shared_ptr<StorageBase<T>>  Tensor<T>::storage() const { throw std::runtime_error("No implement"); }
    template<typename T>
    DimVector Tensor<T>::stride() const { throw std::runtime_error("No implement"); }
	template<typename T>
	idx_type Tensor<T>::stride(idx_type i) const
	{
		throw std::runtime_error("No implement");
	}

    template<typename T>
    void Tensor<T>::sub_(std::shared_ptr<TensorInterface> other)
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    TensorInterfacePtr Tensor<T>::sum() const
    {
        throw std::runtime_error("No implement");
    }
    template<typename T>
    std::string Tensor<T>::to_string() const
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    void transpose_(idx_type dim0, idx_type dim1)
    {
        throw std::runtime_error("No implement");
    }

    template<typename T>
    std::shared_ptr<TensorInterface> transpose(idx_type dim0, idx_type dim1)
    {
        throw std::runtime_error("No implement");
    }
}