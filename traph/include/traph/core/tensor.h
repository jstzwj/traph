#ifndef TRAPH_CORE_TENSOR_H_
#define TRAPH_CORE_TENSOR_H_

#include <algorithm>

#include <traph/core/type.h>
#include <traph/core/index.h>


namespace traph
{
    template<class T>
    class ContiguousStorageBase
    {
    public:
        virtual idx_type size() const = 0;
        virtual size_type element_size() const = 0;

        virtual void resize_(idx_type size) = 0;
    };

    template<class T>
    class TensorBase
    {
    public:
        virtual platform_type platform() = 0;

        virtual device_id device() = 0;

        virtual void reshape(const DimVector& dims) = 0;

        virtual void resize(const DimVector& dims) = 0;

		virtual idx_type offset() const = 0;

		virtual layout_type layout() const = 0;

		virtual DimVector size() const = 0;

		virtual const T* data() const = 0;
		virtual T* data() = 0;

		virtual DimVector strides() const = 0;
    };

    template<class T>
    bool broadcastable(const TensorBase<T> &lhs, const TensorBase<T> & rhs)
    {
        DimVector lhs_dim = lhs.size();
        DimVector rhs_dim = rhs.size();
        if(lhs_dim.size() < 1 || rhs_dim.size() < 1)
            return false;

        idx_type min = std::min(lhs_dim.size(), rhs_dim.size());
        for(idx_type i = 0; i<min;++i)
        {
            if(lhs_dim[i] != rhs_dim[i] &&
                lhs_dim[i] != 1 &&
                rhs_dim[i] != 1)
            {
                return false;
            }
        }

        return true;
    }

    template<class T>
    bool strict_same_shape(const TensorBase<T> &lhs, const TensorBase<T> & rhs)
    {
        DimVector lhs_dim = lhs.size();
        DimVector rhs_dim = rhs.size();
        if(lhs_dim.size() != rhs_dim.size())
            return false;
        for(idx_type i = 0; i < lhs_dim.size(); ++i)
        {
            if(lhs_dim[i] != rhs_dim[i])
                return false;
        }

        return true;
    }
}

#endif