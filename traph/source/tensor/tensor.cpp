#include <traph/tensor/tensor.h>

namespace traph
{
	// definition
    // public
    template<typename T>
    Tensor<T>::Tensor()
        :tensor_impl()
    {
    }

    template<typename T>
    Tensor<T>::Tensor(const DimVector& dimensions)
        : tensor_impl()
    {
		tensor_impl.resize(dimensions);
    }

    template<typename T>
    Tensor<T>::Tensor(const DimVector& dimensions, const DimVector& strides)
        : tensor_impl()
    {
        auto_strides();

        _rep->resize_(_dimensions.flat_size());
    }

    template<typename T>
    Tensor<T>::Tensor(const T& t)
        :_rep(new TensorStorage<T>),
        _dimensions(), _offset(0), _strides()
    {
        _dimensions.resize(1);
        auto_strides();
    }

    template<typename T>
    void Tensor<T>::add_(TensorInterfacePtr other)
    {
		// check tensor other type
        if(other->dtype() != DataType::FLOAT)
            throw std::runtime_error("expected type float tensor");
		// check broadcast.shape = this.shape
        auto shape = broadcast_shape(this->size(), other->size());
        if(shape != this->size())
            throw std::runtime_error("The size of tensor a must match the size of tensor b");
		// ok, get lhs, rhs
		Tensor<T> * lhs = this;
		Tensor<T> * rhs = dynamic_cast<Tensor<T> *>(other.get());
		std::function<void(idx_type, idx_type, idx_type, idx_type)> add_impl =
			[&](idx_type lhs_dim, idx_type rhs_dim, idx_type lhs_idx, idx_type rhs_idx) {

			auto lhs_storage = std::dynamic_pointer_cast<TensorStorage<T>>(lhs->storage())->data_ptr();
			auto rhs_storage = std::dynamic_pointer_cast<TensorStorage<T>>(rhs->storage())->data_ptr();

			idx_type lsh_shape_size = lhs_dim >= -(lhs->size().size())? lhs->size(lhs_dim) : 1;
			idx_type rsh_shape_size = rhs_dim >= -(rhs->size().size()) ? rhs->size(rhs_dim) : 1;
			idx_type max_shape_size = std::max(lsh_shape_size, rsh_shape_size);

			for (idx_type i = 0; i < max_shape_size; ++i)
			{
                if (lhs_dim <= -(lhs->size().size()) && rhs_dim <= -(rhs->size().size()))
                {
                    lhs_storage[lhs_idx] += rhs_storage[rhs_idx];
                }
                else
                {
                    add_impl(lhs_dim - 1, rhs_dim - 1, lhs_idx, rhs_idx);
                }

				if(lsh_shape_size > 1)
					lhs_idx += lhs->stride(lhs_dim);
				if (rsh_shape_size > 1)
					rhs_idx += rhs->stride(rhs_dim);
			}
		};

		add_impl(-1, -1, lhs->offset(), rhs->offset());
    }

    template<typename T>
    void Tensor<T>::apply_(std::function<T(T)> f)
    {
        // sort stride for cache optimization
		DimVector cloned_stride(_strides);
        DimVector sorted_stride(_strides.size());
        for(int i = 0; i<_strides.size(); ++i)
            sorted_stride[i] = i;
        
        for (int i = 0; i < cloned_stride.size() - 1; i++)
            for (int j = 0; j < cloned_stride.size() - 1 - i; j++)
                if (cloned_stride[j] < cloned_stride[j + 1])
                {
                    std::swap(cloned_stride[j], cloned_stride[j+1]);
                    std::swap(sorted_stride[j], sorted_stride[j+1]);
                }
        
        std::function<void(idx_type, idx_type, std::function<T(T)>)> apply_impl =
        [&](idx_type dim_idx, idx_type idx, std::function<T(T)> f){
            idx_type dim = sorted_stride[dim_idx];
            idx_type dim_size = _dimensions.size();

            idx_type step_len = _strides[dim];
            idx_type step_num = _dimensions[dim];
            
            for(idx_type i = 0; i < step_num; ++i)
            {
                if(dim_idx == dim_size - 1)
                    _rep->data[idx] = f(_rep->data[idx]);
                else
                    apply_impl(dim_idx + 1, idx, f);
                idx += step_len;
            }
        };

        if(_dimensions.size() > 0)
            apply_impl(0, _offset, f);
    }

    template<typename T>
    TensorInterfacePtr Tensor<T>::clone() const
    {
        std::shared_ptr<Tensor<T>> cloned_tensor(new Tensor<T>);
        cloned_tensor->_rep = std::dynamic_pointer_cast<TensorStorage<T>>(_rep->clone());
        cloned_tensor->_dimensions = _dimensions;
        cloned_tensor->_offset = _offset;
        cloned_tensor->_strides = _strides;
        
        return cloned_tensor;
    }

    template<typename T>
    void Tensor<T>::cos_()
    {
        apply_([](T a)->f32 {return std::cos(a); });
    }

    template<typename T>
    std::shared_ptr<TensorBase<f32>> Tensor<T>::create_grad()
    {
        return std::shared_ptr<TensorBase<f32>>(new Tensor<f32>(_dimensions));
    }

    template<typename T>
	T* Tensor<T>::data_ptr()
    {
        return _rep->data_ptr();
    }

    template<typename T>
    const T* Tensor<T>::data_ptr() const
    {
        return _rep->data_ptr();
    }

    template<typename T>
    device_id Tensor<T>::device() { return 0; }

    template<typename T>
    DataType Tensor<T>::dtype() const
    {
        return DataType::FLOAT;
    }

    template<typename T>
    bool Tensor<T>::equal(std::shared_ptr<TensorInterface> other) const
    {
        if(other->platform() != this->platform())
            throw std::runtime_error("equal: Two tensors must be the same platform");
        
        if(other->dtype() != this->dtype())
            return false;

        if(other->size() != this->size())
            return false;

        std::shared_ptr<Tensor<T>> other_ptr = std::dynamic_pointer_cast<Tensor<T>>(other);
        
        std::function<bool(idx_type, T*, T*)> equal_impl =
        [&](idx_type dim, T* lhs_idx, T* rhs_idx){
            idx_type dim_size = _dimensions.size();
            
            for(idx_type i = 0; i < _dimensions[dim]; ++i)
            {
                if(dim == dim - 1)
                {
                    if(*lhs_idx != *rhs_idx) return false;
                }
                else
                {
                    if(!equal_impl(dim + 1, lhs_idx, rhs_idx)) return false;
                }
                lhs_idx += _strides[dim];
                rhs_idx += other_ptr->stride(dim);
            }
            return true;
        };

        return equal_impl(0, _rep->data_ptr() + _offset, other_ptr->data_ptr() + other_ptr->offset());
    }

    template<typename T>
	std::shared_ptr<TensorInterface> Tensor<T>::inverse() const
	{
		// FIX ME
		// return std::dynamic_pointer_cast<TensorInterface>(inverse_impl(*this));
		return nullptr;
	}

    template<typename T>
    void Tensor<T>::fill_(T value)
    {
        apply_([&value](T a)->T {return value; });
    }

    template<typename T>
	T Tensor<T>::item() const
    {
        if(_dimensions.flat_size() == 1)
        {
            return _rep->data[_offset];
        }
        else
        {
            throw std::runtime_error("item: only one element tensors can be converted to scalars");
        }
    }

    template<typename T>
	std::shared_ptr<TensorInterface> Tensor<T>::matmul(std::shared_ptr<TensorInterface> mat) const
	{
		auto right_matrix = std::dynamic_pointer_cast<Tensor<T>>(mat);
		return matmul_impl(*this, *right_matrix);
	}

    template<typename T>
    TensorInterfacePtr Tensor<T>::mean() const
    {
        DimVector d(1);
        d[0] = 1;

        TensorPtr<T> result(new Tensor<T>(d));
        auto flat_size = _dimensions.flat_size();
        result->_rep->data[0] = reduce([](T a, T b)->T {return a + b; });
        result->_rep->data[0] /= flat_size;
        return std::dynamic_pointer_cast<TensorInterface>(result);
    }

    template<typename T>
    void Tensor<T>::mul_(T value)
    {
        apply_([value](T a)->T {return a*value; });
    }

    template<typename T>
    void Tensor<T>::mul_(std::shared_ptr<TensorInterface> other)
    {
        // check tensor other type
        if(other->dtype() != DataType::FLOAT)
            throw std::runtime_error("expected type float tensor");
		// check broadcast.shape = this.shape
        auto shape = broadcast_shape(this->size(), other->size());
        if(shape != this->size())
            throw std::runtime_error("The size of tensor a must match the size of tensor b");
		// ok, get lhs, rhs
		Tensor<T> * lhs = this;
		Tensor<T> * rhs = dynamic_cast<Tensor<T> *>(other.get());
		std::function<void(idx_type, idx_type, idx_type, idx_type)> mul_impl =
			[&](idx_type lhs_dim, idx_type rhs_dim, idx_type lhs_idx, idx_type rhs_idx) {

			auto lhs_storage = std::dynamic_pointer_cast<TensorStorage<T>>(lhs->storage())->data_ptr();
			auto rhs_storage = std::dynamic_pointer_cast<TensorStorage<T>>(rhs->storage())->data_ptr();

			idx_type lsh_shape_size = lhs_dim >= -(lhs->size().size())? lhs->size(lhs_dim) : 1;
			idx_type rsh_shape_size = rhs_dim >= -(rhs->size().size()) ? rhs->size(rhs_dim) : 1;
			idx_type max_shape_size = std::max(lsh_shape_size, rsh_shape_size);

			for (idx_type i = 0; i < max_shape_size; ++i)
			{
                if (lhs_dim <= -(lhs->size().size()) && rhs_dim <= -(rhs->size().size()))
                {
                    lhs_storage[lhs_idx] *= rhs_storage[rhs_idx];
                }
                else
                {
                    mul_impl(lhs_dim - 1, rhs_dim - 1, lhs_idx, rhs_idx);
                }

				if(lsh_shape_size > 1)
					lhs_idx += lhs->stride(lhs_dim);
				if (rsh_shape_size > 1)
					rhs_idx += rhs->stride(rhs_dim);
			}
		};

		mul_impl(-1, -1, lhs->offset(), rhs->offset());
    }

    template<typename T>
    idx_type Tensor<T>::ndimension() const
    {
        return _dimensions.size();
    }

    template<typename T>
    void Tensor<T>::neg_()
    {
        apply_([](T a)->T {return -a; });
    }

    template<typename T>
    idx_type Tensor<T>::offset() const { return _offset; }

    template<typename T>
    std::shared_ptr<TensorInterface> Tensor<T>::permute(const DimVector& dims) const
    {
        // check dims
        if(dims.size() != _strides.size())
            throw std::runtime_error("permute dimension must have the same size");
        std::vector<int> check_vec(dims.size(), 0);
        for(int i = 0; i < dims.size();++i)
            if(dims[i] >= 0 && dims[i] < dims.size())
                check_vec[dims[i]] = 1;
            else
                throw std::runtime_error("permute dimension must in ndimension range");
        
        for(int i = 0; i < check_vec.size();++i)
        {
            if(check_vec[i] != 1)
                throw std::runtime_error("permute dimension error");
        }
        // permute
        std::shared_ptr<Tensor<T>> result(new Tensor<T>);
        result->_rep = _rep;
        result->_dimensions = _dimensions;
        result->_offset = _offset;
        result->_strides = _strides;

        for(int i=0; i<dims.size(); ++i)
        {
            result->_dimensions[i] = _dimensions[dims[i]];
            result->_strides[i] = _strides[dims[i]];
        }

        return result;
    }

    template<typename T>
    PlatformType Tensor<T>::platform() const { return PlatformType::CPU; }

    template<typename T>
    void Tensor<T>::pow_(f32 exp)
    {
        apply_([&exp](T a)->T {return std::pow(a, exp); });
    }

    template<typename T>
	T Tensor<T>::reduce(std::function<T(T, T)> f) const
    {
		T result{};
        reduce_impl(result, 0, _offset, f);
        return result;
    }
    
    template<typename T>
    TensorInterfacePtr Tensor<T>::reduce_dim(idx_type dim, std::function<T(T, T)> f) const
    {
        DimVector reduced_dim = _dimensions;
        reduced_dim.erase(dim); // check dim?
        TensorBasePtr<T> result(new Tensor<T>(reduced_dim));
        TensorPtr<T> raw_result = std::dynamic_pointer_cast<Tensor<T>>(result);
        reduce_dim_impl(*(raw_result.get()), 0, dim, _offset, raw_result->_offset, f);
        return std::dynamic_pointer_cast<TensorInterface>(result);
    }
    
    template<typename T>
    void Tensor<T>::reshape_(const DimVector& dims)
    {

    }
    
    template<typename T>
    void Tensor<T>::resize_(const DimVector& dims)
    {
        _dimensions = dims;
        _rep->resize_(dims.flat_size());
        auto_strides();
    }

    template<typename T>
	std::shared_ptr<TensorInterface> Tensor<T>::select(const SliceVector& slice) const
	{
		std::shared_ptr<Tensor<T>> result(new Tensor<T>);
		result->_rep = _rep;

		// dimension
		DimVector dim;
		std::fesetround(FE_TONEAREST);
		for (idx_type i = 0; i < slice.size(); ++i)
		{
			auto& each = slice[i];
			dim.push_back(
				std::lrint(std::ceil((each.end.value_or(_dimensions[i]) - each.start.value_or(0)) / (float)each.step.value_or(1)))
			);
		}
		result->_dimensions = dim;

		// offset
		idx_type new_offset = 1;
		for (idx_type i = 0; i < slice.size(); ++i)
		{
			new_offset *= _strides[i] * slice[i].start.value_or(0);
		}
		result->_offset = _offset + new_offset;

		// strides
		DimVector strides;
		for (idx_type i = 0; i < slice.size(); ++i)
		{
			strides.push_back(_strides[i] * slice[i].step.value_or(1));
		}
		result->_strides = strides;

		return std::dynamic_pointer_cast<TensorInterface>(result);
	}

    template<typename T>
    void Tensor<T>::sin_()
    {
        apply_([](T a)->T {return std::sin(a); });
    }

    template<typename T>
    DimVector Tensor<T>::size() const { return _dimensions;}
	
    template<typename T>
	idx_type Tensor<T>::size(idx_type i) const
	{ 
		auto shape_size = _dimensions.size();
		if (i >= 0 && i < _dimensions.size())
			return _dimensions[i];
		else if (i <= -1 && i >= -_dimensions.size())
			return _dimensions[shape_size + i];
		else
			throw std::runtime_error("Dimension out of range");
	}

    template<typename T>
    DimVector Tensor<T>::stride() const { return _strides; }

    template<typename T>
	idx_type Tensor<T>::stride(idx_type i) const
	{
		auto stride_size = _strides.size();
		if (i >= 0 && i < _strides.size())
			return _strides[i];
		else if (i <= -1 && i >= -_strides.size())
			return _strides[stride_size + i];
		else
			throw std::runtime_error("Stride out of range");
	}

    template<typename T>
    void Tensor<T>::sub_(std::shared_ptr<TensorInterface> other)
    {
        Tensor<T> * lhs = this;
		Tensor<T> * rhs = dynamic_cast<Tensor<T> *>(other.get());
		std::function<void(Tensor<T> *, Tensor<T> *, idx_type, idx_type,idx_type, idx_type)> sub_impl =
			[&](Tensor<T> * lhs, Tensor<T> * rhs, idx_type lhs_dim, idx_type rhs_dim, idx_type lhs_idx, idx_type rhs_idx) {

			auto lhs_storage = std::dynamic_pointer_cast<TensorStorage<T>>(lhs->storage())->data_ptr();
			auto rhs_storage = std::dynamic_pointer_cast<TensorStorage<T>>(rhs->storage())->data_ptr();

			if (lhs_dim < -(lhs->size().size()) && rhs_dim < -(rhs->size().size()))
			{
				lhs_storage[lhs_idx] -= rhs_storage[rhs_idx];
				return;
			}

			idx_type lhs_shape_size = lhs_dim >= -(lhs->size().size())? lhs->size(lhs_dim) : 1;
			idx_type rhs_shape_size = rhs_dim >= -(rhs->size().size()) ? rhs->size(rhs_dim) : 1;
			idx_type max_shape_size = std::max(lhs_shape_size, rhs_shape_size);

			for (idx_type i = 0; i < max_shape_size; ++i)
			{
				sub_impl(lhs, rhs, lhs_dim - 1, rhs_dim - 1, lhs_idx, rhs_idx);

				if(lhs_shape_size > 1)
					lhs_idx += lhs->stride(lhs_dim);
				if (rhs_shape_size > 1)
					rhs_idx += rhs->stride(rhs_dim);
			}
		};

		sub_impl(lhs, rhs, -1, -1, lhs->offset(), rhs->offset());
    }
    
    template<typename T>
    TensorInterfacePtr Tensor<T>::sum() const
    {
        DimVector d(1);
        d[0] = 1;

        TensorPtr<T> result(new Tensor<T>(d));
        result->_rep->data[0] = reduce([](T a, T b)->T {return a + b; });
        return std::dynamic_pointer_cast<TensorInterface>(result);
    }

    template<typename T>
    std::string Tensor<T>::to_string() const
    {
        std::function<std::string(const Tensor<T>&, idx_type, idx_type)> to_string_impl =
			[&](const Tensor<T>& t, idx_type dim, idx_type idx)->std::string {
            std::string result;
			if (dim == t.size().size())
            {
                result += std::to_string(t.data_ptr()[idx]);
				return result;
            }

			for (idx_type i = 0; i < t.size(dim); ++i)
			{
				if (dim != t.size().size() - 1 && i != 0) result += ",\n";
				if(dim != t.size().size() - 1)	result += "[";
				result += to_string_impl(t, dim + 1, idx);
				if (i != t.size(dim) - 1 && dim == t.size().size() - 1)
					result += ",";
				if (dim != t.size().size() - 1) result += "]";

				idx += t.stride(dim);
			}

			return result;
		};

		std::string result;
		result += "[" + to_string_impl(*this, 0, offset()) + "]";
		return result;
    }

    template<typename T>
    void Tensor<T>::transpose_(idx_type dim0, idx_type dim1)
    {
        if(dim0 != dim1 &&
            _dimensions.in_range(dim0) &&
            _dimensions.in_range(dim1))
        {
            std::swap(_dimensions[dim0], _dimensions[dim1]);
            std::swap(_strides[dim0], _strides[dim1]);
        }
    }

    template<typename T>
    std::shared_ptr<TensorInterface> Tensor<T>::transpose(idx_type dim0, idx_type dim1)
    {
        std::shared_ptr<Tensor<T>> result(new Tensor<T>);
        result->_rep = _rep;
        result->_dimensions = _dimensions;
        result->_offset = _offset;
        result->_strides = _strides;

        result->transpose_(dim0, dim1);

        return result;
    }
}