#ifndef TRAPH_NN_MODULE_H_
#define TRAPH_NN_MODULE_H_

#include <memory>
#include <vector>

#include <traph/nn/variable.h>

namespace traph
{
    class Module
    {
    private:
        std::vector<std::shared_ptr<VariableInterface>> parameters;
    public:
        
    };
} // traph

#endif