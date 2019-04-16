#ifndef TRAPH_TENSOR_TENSOR_STORAGE_H_
#define TRAPH_TENSOR_TENSOR_STORAGE_H_

#include<traph/core/tensor_storage.h>

namespace traph
{
    // The real representation of all tensors.
    template<typename T>
    class TensorStorage: public ContiguousStorageBase<T>
    {
    public:
        using value_type = T;
        using self_type = TensorStorage<T>;
        using base_type = ContiguousStorageBase<T>;

        using raw_pointer = self_type*;
        using raw_const_pointer = const self_type*;
        using shared_pointer = std::shared_ptr<self_type>;
        using reference = self_type&;
        using const_reference = const self_type&;

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

        virtual std::shared_ptr<StorageBase<T>> clone() const override
        {
            std::shared_ptr<TensorStorage<T>> cloned_storage(new TensorStorage<T>);
            cloned_storage->data = std::unique_ptr<T[]>(new T[len]);
            std::memcpy(cloned_storage->data.get(), data.get(), len * sizeof(T));
            cloned_storage->len = len;

            return std::dynamic_pointer_cast<StorageBase<T>>(cloned_storage);
        }
        virtual T* data_ptr() override {return data.get();}
        virtual const T* data_ptr() const override {return data.get();}
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

    using DoubleStorage = TensorStorage<f64>;
    using FloatStorage = TensorStorage<f32>;
    using LongStorage = TensorStorage<i64>;
    using IntStorage = TensorStorage<i32>;
    using ShortStorage = TensorStorage<i16>;
    using CharStorage = TensorStorage<i8>;
    using ByteStorage = TensorStorage<u8>;
}

#endif