#ifndef TRAPH_NN_GRAPH_H_
#define TRAPH_NN_GRAPH_H_

#include <utility>
#include <cmath>
#include <string>
#include <vector>

#include <traph/core/type.h>


namespace traph
{
    class FlowGraphNode
    {
    public:

    };

    class FlowGraphEdge
    {
    public:

    };

    class FlowGraph
    {
    private:
        std::vector<FlowGraphNode> _nodes;
        std::vector<FlowGraphEdge> _edges;
    public:
    };
}

#endif