
#include<traph/core/type.h>
#include<traph/core/index.h>
#include<traph/core/utils.h>

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
        std::unique_ptr<TensorStorage<T>> rep;
        DimVector dimensions;
        idx_type offset;
		DimVector strides;
        layout_type order;
    private:
        void auto_strides()
        {
            idx_type dim_num = dimensions.size();
            strides.resize(dim_num);
            size_type stride = 1;
            if(order == layout_type::column_major)
            {
                for(idx_type i = 0; i < dim_num; ++i)
                {
                    strides[i] = stride;
                    stride *= dimensions[i];
                }
            }
            else
            {
                for(idx_type i = dim_num - 1; i >= 0; --i)
                {
                    strides[i] = stride;
                    stride *= dimensions[i];
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
            :rep(new TensorStorage<T>),
            dimensions(), offset(0), strides(), order(layout_type::column_major)
        {
        }

        explicit Tensor(const DimVector& dimensions)
            :rep(new TensorStorage<T>),
            dimensions(dimensions), offset(0), strides(), order(layout_type::column_major)
        {
            auto_strides();
        }

        explicit Tensor(const DimVector& dimensions, layout_type order)
            :rep(new TensorStorage<T>),
            dimensions(dimensions), offset(0), strides(), order(order)
        {
            auto_strides();
        }

        explicit Tensor(const DimVector& dimensions, const DimVector& strides)
            :rep(new TensorStorage<T>),
            dimensions(dimensions), offset(0), strides(strides), order(layout_type::column_major)
        {
            auto_strides();
        }

        explicit Tensor(const DimVector& dimensions, const DimVector& strides, layout_type order)
            :rep(new TensorStorage<T>),
            dimensions(dimensions), offset(0), strides(strides), order(order)
        {
            auto_strides();
        }

        Tensor(const T& t)
            :rep(new TensorStorage<T>),
            dimensions(), offset(0), strides(), order(order)
        {
            dimensions.resize(1);
            auto_strides();
        }

        Tensor(const Tensor& other)
            :rep(new TensorStorage<T>(*other.rep.get())),
            dimensions(other.dimensions),
            offset(other.offset),
            strides(other.strides),
            order(other.order)
        {
        }

        Tensor(Tensor&& other)
            :rep(std::move(other.rep)),
            dimensions(other.dimensions),
            offset(other.offset),
            strides(other.strides),
            order(other.order)
        {
        }

        void reshape(const DimVector& dims)
        {

        }
        // type cast
        DoubleTensor to_double() const
        {
            DoubleTensor result(*this);
            result.rep = result.rep.to_double();
            return result;
        }
        // op
        // index
        T& item()
        {
            if(rep->size() > 0)
            {
                return rep->data[0]; 
            }
            else
            {
                // error
            }
            
        }

        T& index(const DimVector& dims)
        {
            idx_type pos = 0;

            for(idx_type i = 0; i < dimensions.size(); ++i)
            {
                pos += dimensions[i] * strides[i];
            }

            pos += offset;

            return rep->data[pos];
        }

        Tensor& operator ()(idx_type idx)
        {
            
        }

        template<class... Args>
        Tensor& operator ()(idx_type idx, Args... args)
        {
            
        }

        template<class... Args>
        const Tensor& operator ()(idx_type idx, Args... args) const
        {
            
        }
    };
}