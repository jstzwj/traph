#ifndef TRAPH_NN_LAYERS_LOSS
#define TRAPH_NN_LAYERS_LOSS

#include <traph/nn/module.h>

namespace traph
{
    enum class MSELossReduction
    {
        NONE,
        MEAN,
        SUM
    };

    class MSELoss: public Module
    {
    private:
        MSELossReduction _reduction;
    public:
        MSELoss(MSELossReduction reduction = MSELossReduction::MEAN)
            :_reduction(reduction)
        {
        }

        std::shared_ptr<VariableInterface> forward(std::shared_ptr<VariableInterface> input, std::shared_ptr<VariableInterface> target)
        {
            std::shared_ptr<VariableInterface> ret;
            if(_reduction == MSELossReduction::SUM)
            {
                ret = sum(sub(input, target));
            }
            else if(_reduction == MSELossReduction::MEAN)
            {
                // fixme: use mean if it impled
                ret = sum(sub(input, target));
            }
            else
            {
                ret = sum(sub(input, target));
            }
            return ret;
        }
    };
}

#endif // TRAPH_NN_LAYERS_LOSS