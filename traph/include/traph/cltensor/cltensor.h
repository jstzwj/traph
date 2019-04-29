#ifndef TRAPH_CLTENSOR_CLTENSOR_H_
#define TRAPH_CLTENSOR_CLTENSOR_H_

#include <traph/core/type.h>
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
        virtual PlatformType platform() override
        {
            return PlatformType::opencl;
        }

        virtual device_id device() override
        {
            return 0;
        }

        virtual void reshape(const DimVector& dims) override
        {

        }

		virtual idx_type offset() const override
        {

        }

		virtual layout_type layout() const override
        {

        }

		virtual DimVector size() const override
        {

        }

		virtual const T* data() const override
        {

        }
		virtual T* data() override
        {

        }

		virtual DimVector strides() const override
        {
            
        }

    };
}

#endif