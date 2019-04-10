#ifndef TRAPH_TENSOR_SLICE_H_
#define TRAPH_TENSOR_SLICE_H_

#include <utility>
#include <variant>
#include <vector>

#include <traph/core/type.h>
#include <traph/tensor/index.h>
#include <traph/tensor/utils.h>
#include <traph/tensor/tensor.h>

namespace traph
{
    class BasicSlice
    {
    public:
        idx_type start;
        idx_type step;
        idx_type end;
    };

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

    class Slice
    {
    public:
        std::variant<BasicSlice, AdvancedSlice> slice;
        SliceMode mode;
    };

    using SliceVector = std::vector<Slice>;
}

#endif