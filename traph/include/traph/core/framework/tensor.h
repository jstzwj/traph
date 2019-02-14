#ifndef TRAPH_CORE_FRAMEWORK_TENSOR_H_
#define TRAPH_CORE_FRAMEWORK_TENSOR_H_

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace traph
{
    namespace core
    {
        template<typename T>
        class Tensor
        {
        private:
            Eigen::Tensor<double, 4, Eigen::ColMajor> data;
        };
    }
}

#endif // TRAPH_CORE_FRAMEWORK_TENSOR_H_