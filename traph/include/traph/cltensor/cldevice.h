#ifndef TRAPH_CLTENSOR_CLDEVICE_H_
#define TRAPH_CLTENSOR_CLDEVICE_H_

#include <stdexcept>
#include <vector>
#include <iostream>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

namespace traph
{
    class CLDevice
    {
    public:
        CLDevice()
        {
            
        }

		int init()
		{
			
		}
    };
}

#endif