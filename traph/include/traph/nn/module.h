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
        std::string _name;
        std::vector<std::pair<std::string, std::shared_ptr<VariableInterface>>> _parameters;
        std::vector<std::pair<std::string, std::shared_ptr<Module>>> _children;
    public:
        void add_module(const std::string& name, std::shared_ptr<Module> module)
        {
            _children.push_back(std::make_pair(name, module));
        }

        std::vector<std::shared_ptr<Module>> modules()
        {
            std::vector<std::shared_ptr<Module>> result;

            for (const auto &m : _children)
                if(m.second)
                    result.push_back(m.second);

            return result;
        }

        std::vector<std::pair<std::string, std::shared_ptr<VariableInterface>>> named_parameters(bool recurse)
        {
            std::vector<std::pair<std::string, std::shared_ptr<VariableInterface>>> result;
            if(recurse)
            {
                // fixme: children params recurse
                for (const auto &p : _parameters)
                    if(p.first != "")
                        result.push_back(p);
            }
            else
            {
                for (const auto &p : _parameters)
                    if(p.first != "")
                        result.push_back(p);
            }
            return result;
        }

        std::vector<std::shared_ptr<VariableInterface>> parameters(bool recurse=true)
        {
            std::vector<std::shared_ptr<VariableInterface>> result;
            if(recurse)
            {
                // fixme: children params recurse
                for (const auto &p : _parameters)
					if(p.second)
						result.push_back(p.second);

                for (const auto &m : _children)
                {
                    if(m.second)
                    {
                        auto child_params = m.second->parameters(true);
                        for(auto &child_param: child_params)
                            result.push_back(child_param);
                    }
                }
            }
            else
            {
                for (const auto &p : _parameters)
                    result.push_back(p.second);
            }
            return result;
        }

        void register_parameter(const std::string& name, std::shared_ptr<VariableInterface> param)
        {
            _parameters.push_back(std::make_pair(name, param));
        }
    };
} // traph

#endif