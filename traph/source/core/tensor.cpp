#include <traph/core/tensor.h>

namespace traph
{
    bool broadcastable(const DimVector &lhs, const DimVector & rhs)
    {
        if(lhs.size() < 1 || rhs.size() < 1)
            return false;

        idx_type min = std::min(lhs.size(), rhs.size());
		for (idx_type i = -1; i >= -min; --i)
			if (lhs[i] != rhs[i] && lhs[i] != 1 && rhs[i] != 1)
				return false;    

        return true;
    }

	DimVector broadcast_shape(const DimVector &lhs, const DimVector & rhs)
	{
		bool is_broadcastable = broadcastable(lhs, rhs);
		if (!is_broadcastable)
			throw std::runtime_error("The size of tensor a must match the size of tensor b");
		auto max_size = std::max(lhs.size(), rhs.size());
		DimVector result_dim(max_size);

		for (idx_type i = -1; i >= -max_size; --i)
		{
			idx_type lhs_size = i >= -lhs.size() ? lhs[i] : 1;
			idx_type rhs_size = i >= -rhs.size() ? rhs[i] : 1;
			result_dim[max_size + i] = std::max(lhs_size, rhs_size);
		}
		return result_dim;
	}
}