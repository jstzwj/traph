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
        layout_type _order;

    private:
        void auto_strides();

        void apply_impl(idx_type dim, idx_type idx, std::function<T(T)> f);

        void reduce_impl(T& result, idx_type dim, idx_type idx, std::function<T(T,T)> f) const;

        T reduce_dim_kernel(idx_type begin, idx_type step_len, idx_type step_num, std::function<T(T,T)> f) const;

        void reduce_dim_impl(Tensor<T>& result, idx_type dim, idx_type reduce_dim,
            idx_type this_idx, idx_type result_idx,
            std::function<T(T,T)> f) const;
    public:
        Tensor();
        explicit Tensor(const DimVector& dimensions);
        explicit Tensor(const DimVector& dimensions, layout_type order);
        explicit Tensor(const DimVector& dimensions, const DimVector& strides);
        explicit Tensor(const DimVector& dimensions, const DimVector& strides, layout_type order);
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
        virtual void fill_(T value) override;
        virtual std::shared_ptr<TensorInterface> inverse() const override;
        virtual T item() const override;
        virtual std::shared_ptr<TensorInterface> matmul(std::shared_ptr<TensorInterface> mat) const override;
		virtual idx_type offset() const override;
		virtual layout_type order() const override;
        virtual platform_type platform() override;
        virtual T reduce_(std::function<T(T,T)> f) const override;
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
        virtual TensorInterfacePtr sum() const override;
        virtual std::string to_string() const override;
    };

    using DoubleTensor = Tensor<f64>;
    using FloatTensor = Tensor<f32>;
    using LongTensor = Tensor<i64>;
    using IntTensor = Tensor<i32>;
    using ShortTensor = Tensor<i16>;
    using CharTensor = Tensor<i8>;
    using ByteTensor = Tensor<u8>;

	template<typename T>
	using TensorPtr = std::shared_ptr<Tensor<T>>;
	template<typename T>
	using TensorRef = Tensor<T> &;
	template<typename T>
	using TensorConstRef = const Tensor<T>&;

	// definition
    // private
    template<typename T>
    void Tensor<T>::auto_strides()
    {
        idx_type dim_num = _dimensions.size();
        _strides.resize(dim_num);
        idx_type stride = 1;
        if(_order == layout_type::column_major)
        {
            for (idx_type i = dim_num - 1; i >= 0; --i)
            {
                _strides[i] = stride;
                stride *= _dimensions[i];
            }
        }
        else
        {
            for (idx_type i = 0; i < dim_num; ++i)
            {
                _strides[i] = stride;
                stride *= _dimensions[i];
            }
        }
    }

    template<typename T>
    void Tensor<T>::apply_impl(idx_type dim, idx_type idx, std::function<T(T)> f)
    {
        idx_type dim_size = _dimensions.size();

        idx_type step_len = _strides[dim];
        idx_type step_num = _dimensions[dim];
        
        for(idx_type i = 0; i < step_num; ++i)
        {
            if(dim == dim_size - 1)
                _rep->data[idx] = f(_rep->data[idx]);
            else
                apply_impl(dim + 1, idx, f);
            idx += step_len;
        }
    }

    template<typename T>
    void Tensor<T>::reduce_impl(T& result, idx_type dim, idx_type idx, std::function<T(T,T)> f) const
    {
        idx_type dim_size = _dimensions.size();

        idx_type step_len = _strides[dim];
        idx_type step_num = _dimensions[dim];

        for(idx_type i = 0; i < step_num; ++i)
        {
            if(dim == dim_size - 1)
                result = f(result, _rep->data[idx]);
            else
                reduce_impl(result, dim + 1, idx, f);
            idx += step_len;
        }
    }

    template<typename T>
    T Tensor<T>::reduce_dim_kernel(idx_type begin, idx_type step_len, idx_type step_num, std::function<T(T,T)> f) const
    {
        T result{};
        for(idx_type i = 0; i < step_num; ++i)
        {
            result = f(result, _rep->data[begin]);
            begin += step_len;
        }
        return result;
    }

    template<typename T>
    void Tensor<T>::reduce_dim_impl(Tensor<T>& result, idx_type dim, idx_type reduce_dim,
        idx_type this_idx, idx_type result_idx,
        std::function<T(T,T)> f) const
    {
        idx_type dim_size = _dimensions.size();

        if(dim == dim_size)
        {
            result._rep->data[result_idx] = 
                reduce_dim_kernel(this_idx, _strides[reduce_dim], _dimensions[reduce_dim], f);
            return;
        }

        if(dim == reduce_dim)
        {
            reduce_dim_impl(result, dim + 1, reduce_dim, this_idx,result_idx, f);
        }
        else
        {
            for(idx_type i = 0; i < _dimensions[dim]; ++i)
            {
                reduce_dim_impl(result, dim + 1, reduce_dim, this_idx,result_idx, f);
                    
                this_idx += _strides[dim];
                result_idx += result._strides[dim];
            }
        }
    }
    // public
    template<typename T>
    Tensor<T>::Tensor()
        :_rep(new TensorStorage<T>),
        _dimensions(), _offset(0), _strides(), _order(layout_type::column_major)
    {
    }


    template<typename T>
    Tensor<T>::Tensor(const DimVector& dimensions)
        :_rep(new TensorStorage<T>),
        _dimensions(dimensions), _offset(0), _strides(), _order(layout_type::column_major)
    {
        auto_strides();
        
        _rep->resize_(_dimensions.flat_size());
    }

    template<typename T>
    Tensor<T>::Tensor(const DimVector& dimensions, layout_type order)
        :_rep(new TensorStorage<T>),
        _dimensions(dimensions), _offset(0), _strides(), _order(order)
    {
        auto_strides();

        _rep->resize_(_dimensions.flat_size());
    }

    template<typename T>
    Tensor<T>::Tensor(const DimVector& dimensions, const DimVector& strides)
        :_rep(new TensorStorage<T>),
        _dimensions(dimensions), _offset(0), _strides(strides), _order(layout_type::column_major)
    {
        auto_strides();

        _rep->resize_(_dimensions.flat_size());
    }

    template<typename T>
    Tensor<T>::Tensor(const DimVector& dimensions, const DimVector& strides, layout_type order)
        :_rep(new TensorStorage<T>),
        _dimensions(dimensions), _offset(0), _strides(strides), _order(order)
    {
        auto_strides();

        _rep->resize_(_dimensions.flat_size());
    }

    template<typename T>
    Tensor<T>::Tensor(const T& t)
        :_rep(new TensorStorage<T>),
        _dimensions(), _offset(0), _strides(), _order(layout_type::column_major)
    {
        _dimensions.resize(1);
        auto_strides();
    }

    template<typename T>
    void Tensor<T>::add_(TensorInterfacePtr other)
    {
		// check tensor other type

		// check broadcast.shape = this.shape

		// ok, get lhs, rhs
		Tensor<T> * lhs = this;
		Tensor<T> * rhs = dynamic_cast<Tensor<T> *>(other.get());
		std::function<void(Tensor<T> *, Tensor<T> *, idx_type, idx_type,idx_type, idx_type)> add_impl =
			[&](Tensor<T> * lhs, Tensor<T> * rhs, idx_type lhs_dim, idx_type rhs_dim, idx_type lhs_idx, idx_type rhs_idx) {

			auto lhs_storage = std::dynamic_pointer_cast<TensorStorage<T>>(lhs->storage())->data_ptr();
			auto rhs_storage = std::dynamic_pointer_cast<TensorStorage<T>>(rhs->storage())->data_ptr();

			if (lhs_dim < -(lhs->size().size()) && rhs_dim < -(rhs->size().size()))
			{
				lhs_storage[lhs_idx] += rhs_storage[rhs_idx];
				return;
			}

			idx_type lsh_shape_size = lhs_dim >= -(lhs->size().size())? lhs->size(lhs_dim) : 1;
			idx_type rsh_shape_size = rhs_dim >= -(rhs->size().size()) ? rhs->size(rhs_dim) : 1;
			idx_type max_shape_size = std::max(lsh_shape_size, rsh_shape_size);

			for (idx_type i = 0; i < max_shape_size; ++i)
			{
				add_impl(lhs, rhs, lhs_dim - 1, rhs_dim - 1, lhs_idx, rhs_idx);

				if(lsh_shape_size > 1)
					lhs_idx += lhs->stride(lhs_dim);
				if (rsh_shape_size > 1)
					rhs_idx += rhs->stride(rhs_dim);
			}
		};

		add_impl(lhs, rhs, -1, -1, lhs->offset(), rhs->offset());
    }
    template<typename T>
    void Tensor<T>::apply_(std::function<T(T)> f)
    {
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
        cloned_tensor->_order = _order;
        
        return cloned_tensor;
    }
    template<typename T>
    void Tensor<T>::cos_()
    {
        apply_([](T a)->T {return std::cos(a); });
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
    void Tensor<T>::fill_(T value)
    {
        apply_([&value](T a)->T {return value; });
    }

    template<typename T>
    std::shared_ptr<TensorInterface> Tensor<T>::inverse() const
    {
        return std::dynamic_pointer_cast<TensorInterface>(inverse_impl(*this);
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
    idx_type Tensor<T>::offset() const { return _offset; }
    template<typename T>
    layout_type Tensor<T>::order() const { return _order; }
    template<typename T>
    platform_type Tensor<T>::platform() { return platform_type::none; }
    template<typename T>
    T Tensor<T>::reduce_(std::function<T(T,T)> f) const
    {
        T result{};
        reduce_impl(result, 0, _offset, f);
        return result;
    }
    template<typename T>
    TensorInterfacePtr Tensor<T>::reduce_dim(idx_type dim, std::function<T(T,T)> f) const
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
        for(idx_type i = 0; i<slice.size(); ++i)
        {
			auto& each = slice[i];
            dim.push_back(
				std::lrint(std::ceil((each.end.value_or(_dimensions[i]) - each.start.value_or(0))/(float)each.step.value_or(1)))
			);
        }
        result->_dimensions = dim;

        // offset
        idx_type new_offset =1;
        for(idx_type i = 0; i < slice.size(); ++i)
        {
            new_offset *= _strides[i] * slice[i].start.value_or(0);
        }
        result->_offset = _offset + new_offset;

        // strides
        DimVector strides;
        for(idx_type i = 0; i < slice.size(); ++i)
        {
            strides.push_back(_strides[i] * slice[i].step.value_or(1));
        }
        result->_strides = strides;

        result->_order = _order;

        return std::dynamic_pointer_cast<TensorInterface>(result);
    }
    template<typename T>
    void Tensor<T>::sin_()
    {
        apply_([](T a)->T { return std::sin(a); });
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
	std::shared_ptr<StorageBase<T>>  Tensor<T>::storage() const { return _rep; }
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
    TensorInterfacePtr Tensor<T>::sum() const
    {
        DimVector d(1);
        d[0] = 1;

        TensorPtr<T> result(new Tensor<T>(d));
        result->_rep->data[0] = reduce_([](T a, T b)->T {return a + b; });
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
}

#endif // !TRAPH_TENSOR