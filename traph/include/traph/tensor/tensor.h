#ifndef TRAPH_TENSOR
#define TRAPH_TENSOR

#include <initializer_list>
#include <cmath>


#include<traph/core/type.h>
#include<traph/tensor/index.h>
#include<traph/tensor/utils.h>

namespace traph
{
    // The real representation of all tensors.
    template<typename T>
    class TensorStorage
    {
    public:
        using DoubleStorage = TensorStorage<f64>;
        using FloatStorage = TensorStorage<f32>;
        using LongStorage = TensorStorage<i64>;
        using IntStorage = TensorStorage<i32>;
        using ShortStorage = TensorStorage<i16>;
        using CharStorage = TensorStorage<i8>;
        using ByteStorage = TensorStorage<u8>;
        // using HalfStorage = TensorStorage<f16>;
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
        idx_type size() const {return len;}
        size_type element_size() const {return sizeof(T);}

        void resize_(idx_type size)
        {
            if(size < 0 || size == len)
                return;
            idx_type move_size = (size > len ? len: size);
            std::unique_ptr<T[]> temp(new T[size]);
            std::memcpy(temp.get(), data.get(), move_size * sizeof(T));
            data = std::move(temp);

            len = size;
        }

        // type cast
        FloatStorage to_float() const
        {
            FloatStorage result;
            result.resize_(size());
            for(idx_type i = 0; i < size(); ++i)
            {
                result.data[i] = static_cast<f32>(data[i]);
            }
            return result;
        }

        DoubleStorage to_double() const
        {
            DoubleStorage result;
            result.resize_(size());
            for(idx_type i = 0; i < size(); ++i)
            {
                result.data[i] = static_cast<f64>(data[i]);
            }
            return result;
        }

        // fill
        void fill_(T v)
        {
            for(idx_type i = 0; i < size(); ++i)
            {
                data[i] = v;
            }
        }
        /*
        void resize(const DimVector& dimensions)
        {
            idx_type size = 1;
            for(idx_type i = 0; i < dimensions.size(); ++i)
            {
                size *= dimensions[i];
            }

            if(size < 0 || size == len)
                return;
            idx_type move_size = (size > len ? len: size);
            std::unique_ptr<T[]> temp(new idx_type[size]);
            std::memcpy(temp.get(), data.get(), move_size * sizeof(idx_type));
            data = std::move(temp);

            len = size;
        }
        */
    };

    enum layout_type
    {
        row_major,
        column_major
    };

    // ndarray
    template<typename T>
    class Tensor
    {
    private:
        std::unique_ptr<TensorStorage<T>> _rep;
        DimVector _dimensions;
        idx_type _offset;
		DimVector _strides;
        layout_type _order;

        bool _requires_grad;
    private:
        void auto_strides()
        {
            idx_type dim_num = _dimensions.size();
            _strides.resize(dim_num);
            size_type stride = 1;
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
    public:
        using DoubleTensor = Tensor<f64>;
        using FloatTensor = Tensor<f32>;
        using LongTensor = Tensor<i64>;
        using IntTensor = Tensor<i32>;
        using ShortTensor = Tensor<i16>;
        using CharTensor = Tensor<i8>;
        using ByteTensor = Tensor<u8>;
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

        void reshape(const DimVector& dims)
        {

        }
		// info
		idx_type offset() const
		{
			return _offset;
		}

		layout_type layout() const
		{
			return _order;
		}

		DimVector size() const
		{
			return _dimensions;
		}

		const T* data() const
		{
			return _rep->data.get();
		}

		T* data()
		{
			return _rep->data.get();
		}

		DimVector strides() const
		{
			return _strides;
		}

        // type cast
        DoubleTensor to_double() const
        {
            DoubleTensor result(*this);
            result._rep = result._rep.to_double();
            return result;
        }
        // op
        void add_(T value)
        {
            idx_type i = _offset;
            for(idx_type dim = 0;dim < _dimensions.size();++dim)
            {
                for(idx_type step = 0; step < dimension[dim];++step)
                {
                    _rep->data[i] = _rep->data[i] + value;
                    i += _strides[dim];
                }
            }
        }

        void fill_(T value)
        {
			for (idx_type i = _offset; i < _rep->size(); ++i)
			{
				_rep->data[i] = value;
			}
        }

        void abs_()
        {
            idx_type i = _offset;
            for(idx_type dim = 0;dim < _dimensions.size();++dim)
            {
                for(idx_type step = 0; step < dimension[dim];++step)
                {
                    _rep->data[i] = std::abs(_rep->data[i]);
                    i += _strides[dim];
                }
            }
        }
        // index
        T& item()
        {
            if(_rep->size() > 0)
            {
                return _rep->data[0]; 
            }
            else
            {
                // error
            }
            
        }

        T& index(const DimVector& dims)
        {
            idx_type pos = 0;

            for(idx_type i = 0; i < _dimensions.size(); ++i)
            {
                pos += _dimensions[i] * _strides[i];
            }

            pos += _offset;

            return _rep->data[pos];
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