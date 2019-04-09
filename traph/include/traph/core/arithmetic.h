#ifndef TRAPH_ARITHMETIC_H_
#define TRAPH_ARITHMETIC_H_

#include <utility>
#include <cmath>

#include <traph/core/type.h>
#include <traph/core/index.h>
#include <traph/core/utils.h>
#include <traph/core/tensor.h>

namespace traph
{
	template<class T>
	Tensor<T> abs(const Tensor<T> &t)
	{
		Tensor<T> result(t.size());
		idx_type flat_size_end = t.size().flat_size();
		for (idx_type i = t.data_offset(); i < flat_size_end; ++i)
		{
			result.data()[i] = std::abs(t.data()[i]);
		}

		return result;
	}

	template<class T>
	Tensor<T> acos(const Tensor<T> &t)
	{
		Tensor<T> result(t.size());
		idx_type flat_size_end = t.size().flat_size();
		for (idx_type i = t.data_offset(); i < flat_size_end; ++i)
		{
			result.data()[i] = std::acos(t.data()[i]);
		}

		return result;
	}

	template<class T>
	Tensor<T> add(const Tensor<T> &t, T v)
	{
		Tensor<T> result(t.size());
		idx_type flat_size_end = t.size().flat_size();
		for (idx_type i = 0; i < flat_size_end; ++i)
		{
			result.data()[i] = t.data()[i] + v;
		}

		return result;
	}

	template<>
	Tensor<f32> add(const Tensor<f32> &t, f32 v);

	template<>
	Tensor<f64> add(const Tensor<f64> &t, f64 v);

	template<class T>
	Tensor<T> asin(const Tensor<T> &t)
	{
		Tensor<T> result(t.size());
		idx_type flat_size_end = t.size().flat_size();
		for (idx_type i = t.data_offset(); i < flat_size_end; ++i)
		{
			result.data()[i] = std::asin(t.data()[i]);
		}

		return result;
	}

	template<class T>
	Tensor<T> atan(const Tensor<T> &t)
	{
		Tensor<T> result(t.size());
		idx_type flat_size_end = t.size().flat_size();
		for (idx_type i = t.data_offset(); i < flat_size_end; ++i)
		{
			result.data()[i] = std::atan(t.data()[i]);
		}

		return result;
	}

	template<class T>
	void matmul_check(const Tensor<T> &a, const Tensor<T> &b)
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
		// check order
		if (a.layout() != b.layout())
		{
			throw std::runtime_error("matmul: Two matrix shall have the same layout.");
		}
	}

	template<class T>
	Tensor<T> matmul(const Tensor<T> &a, const Tensor<T> &b)
	{
		// check
		matmul_check(a, b);
		// result
		Tensor<T> result = zeros<T>({ a.size()[0], b.size()[1] });
		// compute
		idx_type m = a.size()[0];
		idx_type n = b.size()[1];
		idx_type k = a.size()[1];

		for (idx_type j = 0; j < n; ++j)
		{
			idx_type jm = j * m;
			idx_type jk = j * k;
			for (idx_type i = 0; i < m; ++i)
			{
				idx_type jm_i = jm + i;
				for (idx_type p = 0; p < k; ++p)
				{
					result.data()[jm_i] = a.data()[p*m + i] + b.data()[jk + p];
				}
			}
		}

		return result;
	}

	template<>
	Tensor<f32> matmul(const Tensor<f32> &a, const Tensor<f32> &b);

	template<>
	Tensor<f64> matmul(const Tensor<f64> &a, const Tensor<f64> &b);
}

#endif