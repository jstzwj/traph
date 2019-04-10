#ifndef TENSORA_TENSOR_UTILS_H_
#define TENSORA_TENSOR_UTILS_H_

#include <initializer_list>

#include <traph/core/type.h>
#include <traph/tensor/index.h>

namespace traph
{
    template <class T, idx_type I>
    struct nested_initializer_list
    {
        using type = std::initializer_list<typename nested_initializer_list<T, I - 1>::type>;
    };

    template <class T>
    struct nested_initializer_list<T, 0>
    {
        using type = T;
    };

    template <class T, idx_type I>
    using nested_initializer_list_t = typename nested_initializer_list<T, I>::type;
}

#endif