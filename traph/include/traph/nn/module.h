#ifndef TRAPH_NN_MODULE_H_
#define TRAPH_NN_MODULE_H_

#include <memory>
#include <vector>
#include <map>

#include <traph/nn/variable.h>
#include <traph/nn/operation.h>
#include <traph/nn/parameter.h>
#include <traph/nn/function.h>

namespace traph
{
    class Module
    {
    private:
        std::map<std::string, std::shared_ptr<ParameterInterface>> _parameters;
        std::vector<std::shared_ptr<Module>> _children;
    public:
        std::vector<std::shared_ptr<ParameterInterface>> parameters(bool recurse)
        {
            std::vector<std::shared_ptr<ParameterInterface>> result;
            if(recurse)
            {
                // fixme: children params recurse
                for (const auto &p : _parameters)
                    result.push_back(p.second);
            }
            else
            {
                for (const auto &p : _parameters)
                    result.push_back(p.second);
            }
            return result;
        }

        void register_parameter(const std::string& name, std::shared_ptr<ParameterInterface> param)
        {
            _parameters[name] = param;
        }
    };

    class LinearModule: public Module
    {
    private:
        int _in_features;
        int _out_features;
        std::shared_ptr<VariableInterface> _weight;
        std::shared_ptr<VariableInterface> _bias;
    public:
        LinearModule(int in_features, int out_features, bool bias)
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
} // traph

#endif