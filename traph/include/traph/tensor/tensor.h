#ifndef TRAPH_TENSOR_TENSOR_H_
#define TRAPH_TENSOR_TENSOR_H_

#include <initializer_list>
#include <cmath>
#include <memory>
#include <functional>
#include <stdexcept>


#include<traph/core/type.h>
#include<traph/core/index.h>
#include<traph/core/utils.h>
#include<traph/core/tensor.h>

namespace traph
{
    // The real representation of all tensors.
    template<typename T>
    class TensorStorage: public ContiguousStorageBase<T>
    {
    public:
        using DoubleStorage = TensorStorage<f64>;
        using FloatStorage = TensorStorage<f32>;
        using LongStorage = TensorStorage<i64>;
        using IntStorage = TensorStorage<i32>;
        using ShortStorage = TensorStorage<i16>;
        using CharStorage = TensorStorage<i8>;
        using ByteStorage = TensorStorage<u8>;
    public:
        std::unique_ptr<T[]> data;
        idx_type len;
        TensorStorage()
            :data(nullptr), len(0)
        {
        }

        TensorStorage(const TensorStorage& other)
            :data(new T[other.len]), len(other.len)
        {
            std::memcpy(data.get(), other.data.get(), other.len * sizeof(T));
        }

        TensorStorage(TensorStorage&& other)
            :data(std::move(other.data)), len(other.len)
        {
        }

        TensorStorage& operator=(const TensorStorage& other)
        {
            data = std::make_unique(new T[other.len]);
            std::memcpy(data.get(), other.data.get(), other.len * sizeof(T));
            len = other.len;

            return *this;
        }

        TensorStorage& operator=(TensorStorage&& other)
        {
            data = std::move(other.data);
            len = other.len;

            return *this;
        }

        // size
        virtual idx_type size() const override {return len;}
        virtual size_type element_size() const override {return sizeof(T);}

        virtual void resize_(idx_type size) override
        {
            if(size < 0 || size == len)
                return;
            idx_type move_size = (size > len ? len: size);
            std::unique_ptr<T[]> temp(new T[size]);
            std::memcpy(temp.get(), data.get(), move_size * sizeof(T));
            data = std::move(temp);

            len = size;
        }

        // fill
        virtual void fill_(T v) override
        {
            for(idx_type i = 0; i < size(); ++i)
            {
                data[i] = v;
            }
        }
    };

    // ndarray
    template<typename T>
    class Tensor: public TensorBase<T>
    {
    private:
        std::shared_ptr<TensorStorage<T>> _rep;
        DimVector _dimensions;
        idx_type _offset;
		DimVector _strides;
        layout_type _order;

        bool _requires_grad;
    public:
        using TensorPtr = std::shared_ptr<Tensor<T>>;
        using TensorRef = Tensor<T>&;
        using TensorConstRef = const Tensor<T>&;

        using DoubleTensor = Tensor<f64>;
        using FloatTensor = Tensor<f32>;
        using LongTensor = Tensor<i64>;
        using IntTensor = Tensor<i32>;
        using ShortTensor = Tensor<i16>;
        using CharTensor = Tensor<i8>;
        using ByteTensor = Tensor<u8>;
    private:
        void auto_strides()
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

        void apply_impl(idx_type dim, idx_type idx, std::function<T(T)> f)
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

        void reduce_impl(T& result, idx_type dim, idx_type idx, std::function<T(T,T)> f) const
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

        T reduce_dim_kernel(idx_type begin, idx_type step_len, idx_type step_num, std::function<T(T,T)> f) const
        {
            T result{};
            for(idx_type i = 0; i < step_num; ++i)
            {
                result = f(result, _rep->data[begin]);
                begin += step_len;
            }
            return result;
        }

        void reduce_dim_impl(Tensor<T>& result, idx_type dim, idx_type reduce_dim,
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
    public:
        Tensor()
            :_rep(new TensorStorage<T>),
            _dimensions(), _offset(0), _strides(), _order(layout_type::column_major), _requires_grad(false)
        {
        }

        explicit Tensor(const DimVector& dimensions)
            :_rep(new TensorStorage<T>),
            _dimensions(dimensions), _offset(0), _strides(), _order(layout_type::column_major), _requires_grad(false)
        {
            auto_strides();
			
			_rep->resize_(_dimensions.flat_size());
        }

        explicit Tensor(const DimVector& dimensions, layout_type order)
            :_rep(new TensorStorage<T>),
            _dimensions(dimensions), _offset(0), _strides(), _order(order), _requires_grad(false)
        {
            auto_strides();

			_rep->resize_(_dimensions.flat_size());
        }

        explicit Tensor(const DimVector& dimensions, const DimVector& strides)
            :_rep(new TensorStorage<T>),
            _dimensions(dimensions), _offset(0), _strides(strides), _order(layout_type::column_major), _requires_grad(false)
        {
            auto_strides();

			_rep->resize_(_dimensions.flat_size());
        }

        explicit Tensor(const DimVector& dimensions, const DimVector& strides, layout_type order)
            :_rep(new TensorStorage<T>),
            _dimensions(dimensions), _offset(0), _strides(strides), _order(order), _requires_grad(false)
        {
            auto_strides();

			_rep->resize_(_dimensions.flat_size());
        }

        Tensor(const T& t)
            :_rep(new TensorStorage<T>),
            _dimensions(), _offset(0), strides(), _order(order), _requires_grad(false)
        {
            _dimensions.resize(1);
            auto_strides();
        }

        Tensor(const Tensor& other)
            :_rep(new TensorStorage<T>(*other._rep.get())),
            _dimensions(other._dimensions),
            _offset(other._offset),
            _strides(other._strides),
            _order(other._order),
            _requires_grad(other._requires_grad)
        {
        }

        Tensor(Tensor&& other)
            :_rep(std::move(other._rep)),
            _dimensions(other._dimensions),
            _offset(other._offset),
            _strides(other._strides),
            _order(other._order),
            _requires_grad(other._requires_grad)
        {
        }

        virtual void apply_(std::function<T(T)> f) override
        {
            apply_impl(0, _offset, f);
        }
        virtual void cos_() override
        {
			apply_([](T a)->T {return std::cos(a); });
        }
        virtual TensorBasePtr create_grad() override
        {
            return std::shared_ptr<TensorBase<T>>(new Tensor<T>(_dimensions));
        }
        virtual device_id device() override { return 0; }
        virtual void fill_(T value) override
        {
			apply_([&value](T a)->T {return value; });
        }
        virtual T item() const override
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
		virtual idx_type offset() const override { return _offset; }
		virtual layout_type order() const override { return _order; }
        virtual platform_type platform() override { return platform_type::none; }
        virtual T reduce_(std::function<T(T,T)> f) const override
        {
            T result{0.f};
            reduce_impl(result, 0, _offset, f);
            return result;
        }
        virtual TensorBasePtr reduce_dim(idx_type dim, std::function<T(T,T)> f) const override
        {
            DimVector reduced_dim = _dimensions;
            reduced_dim.erase(dim); // check dim?
            TensorBasePtr result(new Tensor<T>(reduced_dim));
            TensorPtr raw_result = std::dynamic_pointer_cast<Tensor<T>>(result);
			reduce_dim_impl(*(raw_result.get()), 0, dim, _offset, raw_result->_offset, f);
            return result;
        }
        virtual void reshape_(const DimVector& dims) override
        {

        }
        virtual void resize_(const DimVector& dims) override
        {
            _dimensions = dims;
            _rep->resize_(dims.flat_size());
            auto_strides();
        }
        virtual void sin_() override
        {
			apply_([](T a)->T {return std::sin(a); });
        }
		virtual DimVector size() const override { return _dimensions;}
        virtual StorageBase<T>& storage() const override { return *(_rep.get()); }
		virtual DimVector stride() const override { return _strides; }
        virtual TensorBasePtr sum() const override
        {
            DimVector d(1);
            d[0] = 1;

            TensorPtr result(new Tensor<T>(d));
            result->_rep->data[0] = reduce_([](T a, T b)->T {return a + b; });
			return std::dynamic_pointer_cast<TensorBase<T>>(result);
        }
    };

    /*
    template<class T>
    Tensor<T> zeros(std::initializer_list<idx_type> l)
    {
        DimVector dim;
		for (auto i : l)
			dim.push_back(i);

        Tensor<T> result(dim);
        result.fill_(0);

        return result;
    }

    template<class T>
    Tensor<T> ones(std::initializer_list<idx_type> l)
    {
		DimVector dim;
		for (auto i : l)
			dim.push_back(i);

        Tensor<T> result(dim);
        result.fill_(1);

        return result;
    }
    */

    using DoubleTensor = Tensor<f64>;
    using FloatTensor = Tensor<f32>;
    using LongTensor = Tensor<i64>;
    using IntTensor = Tensor<i32>;
    using ShortTensor = Tensor<i16>;
    using CharTensor = Tensor<i8>;
    using ByteTensor = Tensor<u8>;
}

#endif // !TRAPH_TENSOR