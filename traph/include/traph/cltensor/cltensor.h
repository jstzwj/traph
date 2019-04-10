#ifndef TRAPH_CLTENSOR_CLTENSOR_H_
#define TRAPH_CLTENSOR_CLTENSOR_H_

#include <traph/core/tensor.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

namespace traph
{
    template<class T>
    class CLTensorStorage: public ContiguousStorageBase<T>
    {
    public:
        virtual idx_type size() const = 0;
        virtual size_type element_size() const = 0;

        virtual void resize_(idx_type size) = 0;
    };

    template<class T>
    class CLTensor: public TensorBase<T>
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