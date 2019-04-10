#ifndef TRAPH_TENSOR_TENSOR_H_
#define TRAPH_TENSOR_TENSOR_H_

#include <initializer_list>
#include <cmath>
#include <functional>


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

        void apply_dim(idx_type dim, idx_type idx, std::function<T(T)> f)
        {
            idx_type dim_size = _dimensions.size();

            idx_type step_len = _strides[dim];
            idx_type step_num = _dimensions[dim];
            
            for(idx_type i = 0; i < step_num; ++i)
            {
                if(dim == dim_size - 1)
                    _rep->data[idx] = f(_rep->data[idx]);
                else
                    apply_dim(dim + 1, idx, f);
                idx += step_len;
            }
        }

        void reduce_dim(T& result, idx_type dim, idx_type idx, std::function<T(T,T)> f) const
        {
            idx_type dim_size = _dimensions.size();

            idx_type step_len = _strides[dim];
            idx_type step_num = _dimensions[dim];

            for(idx_type i = 0; i < step_num; ++i)
            {
                if(dim == dim_size - 1)
                    result = f(result, _rep->data[idx]);
                else
                    reduce_dim(result, dim + 1, idx, f);
                idx += step_len;
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
            apply_dim(0, _offset, f);
        }
        virtual void cos_() override
        {
			apply_([](T a)->T {return std::cos(a); });
        }
        virtual device_id device() override { return 0; }
        virtual void fill_(T value) override
        {
			apply_([&value](T a)->T {return value; });
        }
		virtual idx_type offset() const override { return _offset; }
		virtual layout_type order() const override { return _order; }
        virtual platform_type platform() override { return platform_type::none; }
        virtual T reduce_(std::function<T(T,T)> f) const override
        {
            T result{0.f};
            reduce_dim(result, 0, _offset, f);
            return result;
        }
        virtual void reshape(const DimVector& dims) override
        {

        }
        virtual void resize(const DimVector& dims) override
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
        virtual T sum() const override
        {
			return reduce_([](T a, T b)->T {return a + b; });
        }
    };

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

    using DoubleTensor = Tensor<f64>;
    using FloatTensor = Tensor<f32>;
    using LongTensor = Tensor<i64>;
    using IntTensor = Tensor<i32>;
    using ShortTensor = Tensor<i16>;
    using CharTensor = Tensor<i8>;
    using ByteTensor = Tensor<u8>;
}

#endif // !TRAPH_TENSOR