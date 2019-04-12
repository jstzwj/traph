#ifndef TRAPH_NN_EXECUTOR_H_
#define TRAPH_NN_EXECUTOR_H_

#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <iterator>
#include <cassert>

#include <traph/core/variable.h>

namespace traph
{
    class Executor
    {
    private:
    public:
        static std::vector<VariableInterface*> topology_sort(VariableInterface* root);
        static std::set<VariableInterface*> collect_backward_tensors(VariableInterface* root);
    };
}


#endif