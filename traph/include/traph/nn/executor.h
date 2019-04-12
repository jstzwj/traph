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
        static void backward(VariableInterface* root)
        {
            
        }

        static std::vector<VariableInterface*> topology_sort(VariableInterface* root)
        {
            std::set<VariableInterface*> all_nodes = collect_backward_tensors(root);
            std::vector<VariableInterface*> visited_nodes;
            int all_size = all_nodes.size();

            for(int i = 0; i<all_size; ++i)
            {
                for(std::set<VariableInterface*>::iterator it = all_nodes.begin(); it != all_nodes.end(); ++it)
                {
                    std::vector<VariableInterfacePtr> cur_inputs = (*it)->inputs();
                    std::vector<VariableInterfacePtr> cur_inputs_no_visited;
                    std::set_difference(cur_inputs.begin(), cur_inputs.end(), visited_nodes.begin(), visited_nodes.end(), 
                            std::inserter(cur_inputs_no_visited, cur_inputs_no_visited.begin()));
                    if(cur_inputs_no_visited.empty())
                    {
                        visited_nodes.push_back(*it);
                        all_nodes.erase(it);
                        break;
                    }
                }
            }

            return visited_nodes;
        }

        static std::set<VariableInterface*> collect_backward_tensors(VariableInterface* root)
        {
            // bfs
            std::set<VariableInterface*> result_set;
            std::list<VariableInterface*> variable_queue;
            variable_queue.push_back(root);
            while(!variable_queue.empty())
            {
                // get first
                VariableInterface* cur = variable_queue.front();
                variable_queue.pop_front();
                // save
                result_set.insert(cur);

                // get inputs and insert into queue then continue
                std::vector<VariableInterfacePtr>& cur_inputs = cur->inputs();
                for(int i = 0; i<cur_inputs.size(); ++i)
                {
                    variable_queue.push_back(cur_inputs[i].get());
                }
            }

            return result_set;
        }
    };
}


#endif