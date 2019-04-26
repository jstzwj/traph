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
} // traph

#endif