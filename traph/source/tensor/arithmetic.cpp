
#include <stdexcept>
#include <algorithm>

#include <traph/tensor/tensor.h>
#include <traph/tensor/arithmetic.h>

#include <eigen3/Eigen/Dense>


#ifdef TRAPH_BUILD_MKL
#include <mkl.h>
#include <mkl_blas.h>
#include <mkl_cblas.h>
#elif defined TRAPH_BUILD_OPENBLAS
#include <openBLAS/cblas.h>
#endif

namespace traph
{
	std::shared_ptr<Tensor<u8>> matmul_impl(const Tensor<u8>& a, const Tensor<u8>& b)
	{
		// check
		matmul_check(a, b);
		// result
		DimVector dim;
		dim.push_back(a.size()[0]);
		dim.push_back(b.size()[1]);
		std::shared_ptr<Tensor<u8>> result(new Tensor<u8>(dim));

		// copy data
		Eigen::Map<const Eigen::Matrix<u8, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_a(a.data_ptr() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<u8, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_b(b.data_ptr() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<u8, Eigen::Dynamic, Eigen::Dynamic> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result->data_ptr());
		return result;
	}

	std::shared_ptr<Tensor<i8>> matmul_impl(const Tensor<i8>& a, const Tensor<i8>& b)
	{
		// check
		matmul_check(a, b);
		// result
		DimVector dim;
		dim.push_back(a.size()[0]);
		dim.push_back(b.size()[1]);
		std::shared_ptr<Tensor<i8>> result(new Tensor<i8>(dim));

		// copy data
		Eigen::Map<const Eigen::Matrix<i8, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_a(a.data_ptr() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<i8, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_b(b.data_ptr() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<i8, Eigen::Dynamic, Eigen::Dynamic> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result->data_ptr());
		return result;
	}

	std::shared_ptr<Tensor<i16>> matmul_impl(const Tensor<i16>& a, const Tensor<i16>& b)
	{
		// check
		matmul_check(a, b);
		// result
		DimVector dim;
		dim.push_back(a.size()[0]);
		dim.push_back(b.size()[1]);
		std::shared_ptr<Tensor<i16>> result(new Tensor<i16>(dim));

		// copy data
		Eigen::Map<const Eigen::Matrix<i16, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_a(a.data_ptr() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<i16, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_b(b.data_ptr() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<i16, Eigen::Dynamic, Eigen::Dynamic> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result->data_ptr());
		return result;
	}

	std::shared_ptr<Tensor<i32>> matmul_impl(const Tensor<i32>& a, const Tensor<i32>& b)
	{
		// check
		matmul_check(a, b);
		// result
		DimVector dim;
		dim.push_back(a.size()[0]);
		dim.push_back(b.size()[1]);
		std::shared_ptr<Tensor<i32>> result(new Tensor<i32>(dim));

		// copy data
		Eigen::Map<const Eigen::Matrix<i32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_a(a.data_ptr() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<i32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_b(b.data_ptr() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<i32, Eigen::Dynamic, Eigen::Dynamic> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result->data_ptr());
		return result;
	}

	std::shared_ptr<Tensor<i64>> matmul_impl(const Tensor<i64>& a, const Tensor<i64>& b)
	{
		// check
		matmul_check(a, b);
		// result
		DimVector dim;
		dim.push_back(a.size()[0]);
		dim.push_back(b.size()[1]);
		std::shared_ptr<Tensor<i64>> result(new Tensor<i64>(dim));

		// copy data
		Eigen::Map<const Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_a(a.data_ptr() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_b(b.data_ptr() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<i64, Eigen::Dynamic, Eigen::Dynamic> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result->data_ptr());
		return result;
	}

	std::shared_ptr<Tensor<f32>> matmul_impl(const Tensor<f32>& a, const Tensor<f32>& b)
	{
		// check
		matmul_check(a, b);
		// result
		DimVector dim;
		dim.push_back(a.size()[0]);
		dim.push_back(b.size()[1]);
		std::shared_ptr<Tensor<f32>> result(new Tensor<f32>(dim));

		

#ifdef TRAPH_BUILD_EIGEN
		// copy data
		Eigen::Map<const Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> eigen_a(a.data_ptr() + a.offset(), a.size()[0], a.size()[1], Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(a.stride(1), a.stride(0)));
		Eigen::Map<const Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> eigen_b(b.data_ptr() + b.offset(), b.size()[0], b.size()[1], Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(b.stride(1), b.stride(0)));

		Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result->data_ptr());
#elif defined TRAPH_BUILD_MKL
		CBLAS_LAYOUT a_layout = a.order() == layout_type::column_major ? CBLAS_LAYOUT::CblasColMajor : CBLAS_LAYOUT::CblasRowMajor;

		cblas_sgemm(a_layout,
			CBLAS_TRANSPOSE::CblasNoTrans,
			CBLAS_TRANSPOSE::CblasNoTrans,
			a.size()[0],
			b.size()[1],
			a.size()[1],
			1.f,
			a.data_ptr(),
			a.size()[0],
			b.data_ptr(),
			b.size()[0],
			0.f,
			result->data_ptr(),
			result->size()[0]);
#endif
		return result;
	}

	std::shared_ptr<Tensor<f64>> matmul_impl(const Tensor<f64>& a, const Tensor<f64>& b)
	{
		// check
		matmul_check(a, b);
		// result
		DimVector dim;
		dim.push_back(a.size()[0]);
		dim.push_back(b.size()[1]);
		std::shared_ptr<Tensor<f64>> result(new Tensor<f64>(dim));

#ifdef TRAPH_BUILD_EIGEN
		// copy data
		Eigen::Map<const Eigen::Matrix<f64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_a(a.data_ptr() + a.offset(), a.size()[0], a.size()[1]);
		Eigen::Map<const Eigen::Matrix<f64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_b(b.data_ptr() + b.offset(), b.size()[0], b.size()[1]);

		Eigen::Matrix<f64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_c = eigen_a * eigen_b;
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * b.size()[1], result->data_ptr());
#elif defined TRAPH_BUILD_MKL
		CBLAS_LAYOUT a_layout = a.order() == layout_type::column_major ? CBLAS_LAYOUT::CblasColMajor : CBLAS_LAYOUT::CblasRowMajor;

		cblas_dgemm(a_layout,
			CBLAS_TRANSPOSE::CblasNoTrans,
			CBLAS_TRANSPOSE::CblasNoTrans,
			a.size()[0],
			b.size()[1],
			a.size()[1],
			1.f,
			a.data_ptr(),
			a.size()[0],
			b.data_ptr(),
			b.size()[0],
			0.f,
			result->data_ptr(),
			result->size()[0]);
#endif

		return result;
	}

	std::shared_ptr<Tensor<f32>> inverse_impl(const Tensor<f32>& a)
	{
		// result
		std::shared_ptr<Tensor<f32>> result(new Tensor<f32>(a.size()[0], a.size()[1]));

		// copy data
		Eigen::Map<const Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_a(a.data_ptr() + a.offset(), a.size()[0], a.size()[1]);

		Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_c = eigen_a.inverse();
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * a.size()[1], result->data_ptr());
		return result;
	}

	std::shared_ptr<Tensor<f64>> inverse_impl(const Tensor<f64>& a)
	{
		// result
		std::shared_ptr<Tensor<f64>> result(new Tensor<f64>(a.size()[0], a.size()[1]));

		// copy data
		Eigen::Map<const Eigen::Matrix<f64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_a(a.data_ptr() + a.offset(), a.size()[0], a.size()[1]);

		Eigen::Matrix<f64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_c = eigen_a.inverse();
		// copy to result
		std::copy(eigen_c.data(), eigen_c.data() + a.size()[0] * a.size()[1], result->data_ptr());
		return result;
	}
}
