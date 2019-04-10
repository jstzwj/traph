#ifndef TRAPH_CLTENSOR_CLCONTEXT_H_
#define TRAPH_CLTENSOR_CLCONTEXT_H_

#include <stdexcept>
#include <cstring>
#include <vector>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

namespace traph
{
    class CLContext
    {
    private:
        cl_context _context;
    public:
        CLContext(cl_context_properties* cprops)
            :_context(nullptr)
        {
            cl_int status = 0;
            _context = clCreateContextFromType(
                        cprops,
                        CL_DEVICE_TYPE_GPU,
                        NULL,
                        NULL,
                        &status);
            if (status != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Creating Context.(clCreateContexFromType)\n");
            }
        }

        std::size_t device_num()
        {
            if(_context == nullptr)
                return 0;
            cl_int status = 0;
            std::size_t deviceListSize = 0;
            status = clGetContextInfo(_context,
                          CL_CONTEXT_DEVICES,
                          0,
                          NULL,
                          &deviceListSize);
            if (status != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Getting Context Info device list size, clGetContextInfo)\n");
            }
            return deviceListSize;
        }

        std::vector<cl_device_id> get_device_list()
        {
            if(_context == nullptr)
                return {};

            cl_int status = 0;
            auto deviceListSize = device_num();
            std::vector<cl_device_id> devices(deviceListSize);
            status = clGetContextInfo(_context,
                                    CL_CONTEXT_DEVICES,
                                    deviceListSize,
                                    devices.data(),
                                    NULL);
            if (status != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Getting Context Info (device list, clGetContextInfo)\n");
            }

            return devices;
        }

        cl_program create_program(const char* kernelSourceCode)
        {
            if(_context == nullptr)
                return {};

            cl_int status = 0;

            size_t sourceSize[] = {strlen(kernelSourceCode)};
            cl_program program = clCreateProgramWithSource(_context,
                                1,
                                &kernelSourceCode,
                                sourceSize,
                                &status);
            if (status != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Loading Binary into cl_program (clCreateProgramWithBinary)\n");
            }

            return program;
        }

        cl_mem create_buffer(std::size_t size)
        {
            if(_context == nullptr)
                return {};

            cl_int status = 0;

            cl_mem outputBuffer = clCreateBuffer(
									_context,
									CL_MEM_ALLOC_HOST_PTR,
									size,
									NULL,
									&status);
            if (status != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Create Buffer, outputBuffer. (clCreateBuffer)\n");
            }

            return outputBuffer;
        }
    };
}

#endif