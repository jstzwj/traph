#ifndef TRAPH_TENSOR_SLICE_H_
#define TRAPH_TENSOR_SLICE_H_

#include <vector>
#include <optional>

#include <traph/core/type.h>
#include <traph/core/index.h>
#include <traph/core/utils.h>

namespace traph
{
    class BasicSlice
    {
    public:
        std::optional<idx_type> start;
		std::optional<idx_type> step;
		std::optional<idx_type> end;
    };

    /*
    class AdvancedSlice
    {
    public:
        std::vector<idx_type> indices;
    };

    enum SliceMode
    {
        BASIC,
        ADVANCED
    };
    */

    class Slice
    {
    public:
		std::optional<idx_type> start;
		std::optional<idx_type> step;
		std::optional<idx_type> end;

        Slice()
			:start(), step(), end()
		{
		}

        Slice(idx_type start, idx_type end)
			:start(start), step(), end(end)
		{
		}

		Slice(idx_type start, idx_type end, idx_type step)
			:start(start), end(end), step(step)
		{
		}
    };

    using SliceVector = std::vector<Slice>;
}

#endif