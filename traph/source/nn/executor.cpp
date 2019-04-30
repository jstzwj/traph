#include <traph/nn/executor.h>

#include <map>

namespace traph
{
    std::vector<VariableInterface*> Executor::topology_sort(VariableInterface* root)
    {
        std::set<VariableInterface*> all_nodes = collect_backward_tensors(root);
		// std::set<VariableInterface*> visited_nodes;
        std::vector<VariableInterface*> sort_result;
		std::map<VariableInterface*, int> indegrees;
        std::size_t all_size = all_nodes.size();

		for (std::set<VariableInterface*>::iterator it = all_nodes.begin(); it != all_nodes.end(); ++it)
			indegrees[*it] = 0;

		for (std::set<VariableInterface*>::iterator it = all_nodes.begin(); it != all_nodes.end(); ++it)
		{
			std::vector<VariableInterfacePtr>& cur_inputs = (*it)->inputs();
			for (auto &each : cur_inputs)
				if(indegrees.find(each.get()) != indegrees.end())
					indegrees[each.get()]++;
		}

        for(int i = 0; i<all_size; ++i)
        {
            for(std::set<VariableInterface*>::iterator it = all_nodes.begin(); it != all_nodes.end(); ++it)
            {
                std::vector<VariableInterfacePtr> cur_inputs = (*it)->inputs();
				if (indegrees[*it] == 0)
				{
					sort_result.push_back(*it);
					all_nodes.erase(it);
					for (auto& each:cur_inputs)
					{
						if (indegrees.find(each.get()) != indegrees.end())
							indegrees[each.get()]--;
					}
					break;
				}
            }
        }

		return sort_result;
    }

    std::set<VariableInterface*> Executor::collect_backward_tensors(VariableInterface* root)
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
            std::vector<VariableInterfacePtr>& cur_inputs(cur->inputs());
            for(int i = 0; i<cur_inputs.size(); ++i)
            {
                if(cur_inputs[i]->requires_grad())
                    variable_queue.push_back(cur_inputs[i].get());
            }
        }

        return result_set;
    }
}