#ifndef TRAPH_NN_OPTIM_H_
#define TRAPH_NN_OPTIM_H_

#include <memory>
#include <vector>

#include <traph/nn/parameter.h>

namespace traph
{
    class Optimizer
    {
    protected:
        std::vector<std::shared_ptr<VariableInterface>> _params;
    public:
        Optimizer(std::vector<std::shared_ptr<VariableInterface>> params)
            :_params(params)
        {
        }

        virtual void step() = 0;

        void zero_grad()
        {
            for(auto& each_param: _params)
            {
				if(each_param->grad())
					each_param->grad()->fill_(0);
            }
        }
    };

    class SGD:public Optimizer
    {
    private:
        float _lr;
    public:
        SGD(std::vector<std::shared_ptr<VariableInterface>> params, 
            float lr, float momentum=0, float dampening=0, float weight_decay=0,
            bool nesterov=false)
            :Optimizer(params), _lr(lr)
        {
        }

        virtual void step() override
        {
            for(auto& each:_params)
            {
                auto d_p = each->grad();

                auto cloned_d_p = std::dynamic_pointer_cast<TensorBase<f32>>(d_p->clone());
                cloned_d_p->mul_(_lr);
                each->data()->add_(cloned_d_p);
            }
        }
    };
}

#endif