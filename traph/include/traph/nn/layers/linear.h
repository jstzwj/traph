#ifndef TRAPH_NN_LAYERS_LINEAR
#define TRAPH_NN_LAYERS_LINEAR


#include <traph/nn/module.h>

namespace traph
{
    class Linear: public Module
    {
    private:
        int _in_features;
        int _out_features;
        std::shared_ptr<VariableInterface> _weight;
        std::shared_ptr<VariableInterface> _bias;
    public:
        Linear(int in_features, int out_features, bool bias)
        {
            _in_features = in_features;
            _out_features = out_features;
            _weight = std::shared_ptr<VariableInterface>(new FloatParameter({out_features, in_features}));
            if(bias)
                _bias = std::shared_ptr<VariableInterface>(new FloatParameter({out_features}));
            
            register_parameter("weight", std::dynamic_pointer_cast<FloatParameter>(_weight));
            register_parameter("bias", std::dynamic_pointer_cast<FloatParameter>(_bias));
        }

        std::shared_ptr<VariableInterface> forward(std::shared_ptr<VariableInterface> input)
        {
            std::shared_ptr<VariableInterface> result;
            if(_bias)
                result = add(matmul(input, transpose(_weight, 0, 1)), _bias);
            else
                result = matmul(input, transpose(_weight, 0, 1));
            
            return result;
        }
    };
}

#endif // TRAPH_NN_LAYERS_LINEAR