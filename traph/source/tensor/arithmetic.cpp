
#include <stdexcept>
#include <algorithm>

#include <traph/tensor/arithmetic.h>

#include <eigen3/Eigen/Dense>

#ifdef TRAPH_BUILD_OPENBLAS
#include <traph/core/openblas_backend.h>
#endif

#ifdef TRAPH_BUILD_MKL
#include <mkl.h>
#include <mkl_blas.h>
#include <mkl_cblas.h>
#endif

namespace traph
{
	Tensor<f32> add(const Tensor<f32> &t, f32 v)
	{
		Tensor<f32> result(t.size());
#ifdef TRAPH_BUILD_MKL
		result.fill_(v);
		cblas_saxpy(t.size().flat_size(), 1.f, t.data(), 1, result.data(), 1);
#endif
		return result;
	}

	Tensor<f64> add(const Tensor<f64> &t, f64 v)
	{
		Tensor<f64> result(t.size());
#ifdef TRAPH_BUILD_MKL
		result.fill_(v);
		cblas_daxpy(t.size().flat_size(), 1.f, t.data(), 1, result.data(), 1);
#endif
		return result;
	}

	Tensor<u8> matmul(const Tensor<u8> &a, const Tensor<u8> &b)
	{
		// check
		matmul_check(a, b);
		// result
		Tensor<u8> result = zeros<u8>({ a.size()[0], b.size()[1] });

		// copy data
		Eigen::Map<const Eigen::Matrix<u8, Eigen::Dynamic, Eigen::Dynamic>> eigen_a(a.data() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<u8, Eigen::Dynamic, Eigen::Dynamic>> eigen_b(b.data() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<u8, Eigen::Dynamic, Eigen::Dynamic> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result.data());
		return result;
	}

	Tensor<i8> matmul(const Tensor<i8> &a, const Tensor<i8> &b)
	{
		// check
		matmul_check(a, b);
		// result
		Tensor<i8> result = zeros<i8>({ a.size()[0], b.size()[1] });

		// copy data
		Eigen::Map<const Eigen::Matrix<i8, Eigen::Dynamic, Eigen::Dynamic>> eigen_a(a.data() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<i8, Eigen::Dynamic, Eigen::Dynamic>> eigen_b(b.data() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<i8, Eigen::Dynamic, Eigen::Dynamic> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result.data());
		return result;
	}

	Tensor<i16> matmul(const Tensor<i16> &a, const Tensor<i16> &b)
	{
		// check
		matmul_check(a, b);
		// result
		Tensor<i16> result = zeros<i16>({ a.size()[0], b.size()[1] });

		// copy data
		Eigen::Map<const Eigen::Matrix<i16, Eigen::Dynamic, Eigen::Dynamic>> eigen_a(a.data() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<i16, Eigen::Dynamic, Eigen::Dynamic>> eigen_b(b.data() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<i16, Eigen::Dynamic, Eigen::Dynamic> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result.data());
		return result;
	}

	Tensor<i32> matmul(const Tensor<i32> &a, const Tensor<i32> &b)
	{
		// check
		matmul_check(a, b);
		// result
		Tensor<i32> result = zeros<i32>({ a.size()[0], b.size()[1] });

		// copy data
		Eigen::Map<const Eigen::Matrix<i32, Eigen::Dynamic, Eigen::Dynamic>> eigen_a(a.data() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<i32, Eigen::Dynamic, Eigen::Dynamic>> eigen_b(b.data() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<i32, Eigen::Dynamic, Eigen::Dynamic> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result.data());
		return result;
	}

	Tensor<i64> matmul(const Tensor<i64> &a, const Tensor<i64> &b)
	{
		// check
		matmul_check(a, b);
		// result
		Tensor<i64> result = zeros<i64>({ a.size()[0], b.size()[1] });

		// copy data
		Eigen::Map<const Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic>> eigen_a(a.data() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic>> eigen_b(b.data() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result.data());
		return result;
	}

	Tensor<f32> matmul(const Tensor<f32> &a, const Tensor<f32> &b)
	{
		// check
		matmul_check(a, b);
		// result
		Tensor<f32> result = zeros<f32>({ a.size()[0], b.size()[1] });

	#ifdef TRAPH_BUILD_MKL
		CBLAS_LAYOUT a_layout = a.layout() == layout_type::column_major ? CBLAS_LAYOUT::CblasColMajor : CBLAS_LAYOUT::CblasRowMajor;

		cblas_sgemm(a_layout,
			CBLAS_TRANSPOSE::CblasNoTrans,
			CBLAS_TRANSPOSE::CblasNoTrans,
			a.size()[0],
			b.size()[1],
			a.size()[1],
			1.f,
			a.data(),
			a.size()[0],
			b.data(),
			b.size()[0],
			0.f,
			result.data(),
			result.size()[0]);
	#endif
		return result;
	}

	Tensor<f64> matmul(const Tensor<f64> &a, const Tensor<f64> &b)
	{
		// check
		matmul_check(a, b);
		// result
		Tensor<f64> result = zeros<f64>({ a.size()[0], b.size()[1] });

#ifdef TRAPH_BUILD_MKL
		CBLAS_LAYOUT a_layout = a.layout() == layout_type::column_major ? CBLAS_LAYOUT::CblasColMajor : CBLAS_LAYOUT::CblasRowMajor;

		cblas_dgemm(a_layout,
			CBLAS_TRANSPOSE::CblasNoTrans,
			CBLAS_TRANSPOSE::CblasNoTrans,
			a.size()[0],
			b.size()[1],
			a.size()[1],
			1.f,
			a.data(),
			a.size()[0],
			b.data(),
			b.size()[0],
			0.f,
			result.data(),
			result.size()[0]);
#endif

		return result;
	}
}
