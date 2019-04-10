#ifndef TRAPH_CUTENSOR_CUTENSOR_H_
#define TRAPH_CUTENSOR_CUTENSOR_H_

#include <cuda.h>

namespace traph
{

    // The real representation of all tensors.
    template<typename T>
    class CUTensorStorage
    {
    public:
        using CUDoubleStorage = CUTensorStorage<f64>;
        using CUFloatStorage = CUTensorStorage<f32>;
        using CULongStorage = CUTensorStorage<i64>;
        using CUIntStorage = CUTensorStorage<i32>;
        using CUShortStorage = CUTensorStorage<i16>;
        using CUCharStorage = CUTensorStorage<i8>;
        using CUByteStorage = CUTensorStorage<u8>;
        // using CUHalfStorage = CUTensorStorage<f16>;
    public:
        CUTensorStorage()
        {
        }

        CUTensorStorage(const CUTensorStorage& other)
        {
        }

        CUTensorStorage(CUTensorStorage&& other)
        {
        }

        CUTensorStorage& operator=(const CUTensorStorage& other)
        {
        }

        CUTensorStorage& operator=(CUTensorStorage&& other)
        {
        }

        // size
        idx_type size() const {}
        size_type element_size() const {}

        void resize_(idx_type size)
        {
        }

        // type cast
        FloatStorage to_float() const
        {
        }

        DoubleStorage to_double() const
        {
        }
    };

    class CUTensor
    {

    };
}

#endif