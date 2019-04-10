#ifndef TRAPH_CLTENSOR_CLTENSOR_H_
#define TRAPH_CLTENSOR_CLTENSOR_H_

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

namespace traph
{
    class CLTensor
    {

    };
}

#endif