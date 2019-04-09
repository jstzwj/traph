
#include<stdexcept>

#include <traph/core/arithmetic.h>

#ifdef TRAPH_BUILD_OPENBLAS
#include<traph/core/openblas_backend.h>
#endif

#ifdef TRAPH_BUILD_MKL
#include<mkl.h>
#include<mkl_blas.h>
#include<mkl_cblas.h>
#endif

namespace traph
{
	template<>
	Tensor<f32> add(const Tensor<f32> &t, f32 v)
	{
		Tensor<f32> result(t.size());
#ifdef TRAPH_BUILD_MKL
		result.fill_(v);
		cblas_saxpy(t.size().flat_size(), 1.f, t.data(), 1, result.data(), 1);
#endif
		return result;
	}

	template<>
	Tensor<f64> add(const Tensor<f64> &t, f64 v)
	{
		Tensor<f64> result(t.size());
#ifdef TRAPH_BUILD_MKL
		result.fill_(v);
		cblas_daxpy(t.size().flat_size(), 1.f, t.data(), 1, result.data(), 1);
#endif
		return result;
	}

    template<>
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

	template<>
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
