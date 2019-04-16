#ifndef TRAPH_CORE_TENSOR_STORAGE_H_
#define TRAPH_CORE_TENSOR_STORAGE_H_
namespace traph
{
    template<class T>
    class StorageBase
    {
    public:
        using value_type = T;
        using self_type = StorageBase<T>;

        using raw_pointer = self_type*;
        using raw_const_pointer = const self_type*;
        using shared_pointer = std::shared_ptr<self_type>;
        using reference = self_type&;
        using const_reference = const self_type&;
    public:
        virtual std::shared_ptr<StorageBase<T>> clone() const = 0;
        virtual T* data_ptr() = 0;
        virtual const T* data_ptr() const = 0;
        virtual size_type element_size() const = 0;
        virtual void fill_(T v) = 0;
        virtual void resize_(idx_type size) = 0;
        virtual idx_type size() const = 0;
    };

    template<class T>
    class ContiguousStorageBase: public StorageBase<T>
    {
    public:
        using value_type = T;
        using self_type = ContiguousStorageBase<T>;
        using base_type = StorageBase<T>;

        using raw_pointer = self_type*;
        using raw_const_pointer = const self_type*;
        using shared_pointer = std::shared_ptr<self_type>;
        using reference = self_type&;
        using const_reference = const self_type&;
    public:
        virtual std::shared_ptr<StorageBase<T>> clone() const = 0;
        virtual T* data_ptr() = 0;
        virtual const T* data_ptr() const = 0;
        virtual size_type element_size() const = 0;
        virtual void fill_(T v) = 0;
        virtual void resize_(idx_type size) = 0;
        virtual idx_type size() const = 0;
    };
}


#endif