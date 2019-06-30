#ifndef TRAPH_TENSOR_ARITHMETIC_H_
#define TRAPH_TENSOR_ARITHMETIC_H_

#include <utility>
#include <cmath>
#include <memory>

#include <traph/core/type.h>
#include <traph/core/index.h>
#include <traph/core/utils.h>
#include <traph/tensor/tensor.h>

namespace traph
{
	template<typename T>
	class Tensor;

	template<class T>
	void matmul_check(const Tensor<T>& a, const Tensor<T>& b)
	{
		// check dimension
		if (a.size().size() > 2 || b.size().size() > 2)
		{
			throw std::runtime_error("matmul: Two parameters shall be matrix (2D).");
		}
		// check a[1]  and b[0]
		if (a.size()[1] != b.size()[0])
		{
			throw std::runtime_error("matmul: Dimension 0 of the first matrix shall be equal to dimension 1 of the second matrix.");
		}
	}

	std::shared_ptr<Tensor<u8>> matmul_impl(const Tensor<u8>& a, const Tensor<u8>& b);

	std::shared_ptr<Tensor<i8>> matmul_impl(const Tensor<i8>& a, const Tensor<i8>& b);

	std::shared_ptr<Tensor<i16>> matmul_impl(const Tensor<i16>& a, const Tensor<i16>& b);

	std::shared_ptr<Tensor<i32>> matmul_impl(const Tensor<i32>& a, const Tensor<i32>& b);

	std::shared_ptr<Tensor<i64>> matmul_impl(const Tensor<i64>& a, const Tensor<i64>& b);

	std::shared_ptr<Tensor<f32>> matmul_impl(const Tensor<f32>& a, const Tensor<f32>& b);

	std::shared_ptr<Tensor<f64>> matmul_impl(const Tensor<f64>& a, const Tensor<f64>& b);

	std::shared_ptr<Tensor<f32>> inverse_impl(const Tensor<f32>& a);

	std::shared_ptr<Tensor<f64>> inverse_impl(const Tensor<f64>& a);
}

#endif