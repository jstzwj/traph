#ifndef TRAPH_CORE_TENSOR_H_
#define TRAPH_CORE_TENSOR_H_

#include <traph/core/type.h>

namespace traph
{
    class ContiguousStorageBase
    {
    public:
        virtual idx_type size() const = 0;
        virtual size_type element_size() const = 0;

        virtual void resize_(idx_type size) = 0;
    };

    class TensorBase
    {
    public:
        virtual void reshape(const DimVector& dims) = 0;

		virtual idx_type offset() const = 0;

		virtual layout_type layout() const = 0;

		virtual DimVector size() const = 0;

		virtual const T* data() const = 0;
		virtual T* data() = 0;

		virtual DimVector strides() const = 0;

    };
}

#endif