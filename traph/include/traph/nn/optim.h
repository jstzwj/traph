#ifndef TRAPH_NN_OPTIM_H_
#define TRAPH_NN_OPTIM_H_

#include <memory>
#include <vector>

#include <traph/nn/parameter.h>

namespace traph
{
    class Optimizer
    {
    private:
        std::vector<std::shared_ptr<ParameterInterface>> _params;
    public:
        Optimizer(std::vector<std::shared_ptr<ParameterInterface>> params)
            :_params(params)
        {
        }

        virtual void step() = 0;

        void zero_grad()
        {
        }
    };
}

#endif