#ifndef TRAPH_NN_OPERATION_H_
#define TRAPH_NN_OPERATION_H_

#include <utility>
#include <cmath>
#include <string>

#include <traph/core/type.h>
#include <traph/core/index.h>
#include <traph/core/utils.h>
#include <traph/core/variable.h>
#include <traph/tensor/tensor.h>
#include <traph/nn/variable.h>

namespace traph
{
    template<class T>
    class OperationBase
    {
    public:
        virtual name() = 0;
    };

    template<class T>
    class OperationOneParam
    {
    public:
        virtual name() = 0;
    };

    template<class T>
    class OperationTwoParam
    {
    public:
        virtual name() = 0;
    };

    template<class T>
    class OperationThreeParam
    {
    public:
        virtual name() = 0;
    };
}

#endif