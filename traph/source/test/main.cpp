#include <traph/core/tensor.h>

int main()
{
    traph::Tensor<float> a = traph::zeros({2, 2});
    traph::Tensor<float> w = traph::ones({3, 2});
    traph::Tensor<float> result = traph::matmul(w, a);
    result.backward();
    return 0;
}