#ifndef TRAPH_CLTENSOR_CLPLATFORM_H_
#define TRAPH_CLTENSOR_CLPLATFORM_H_

#include <stdexcept>
#include <vector>


#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

namespace traph
{
    class CLPlatform
    {
    public:
        CLPlatform()
        {

        }

        int platform_num()
        {
            cl_int status = 0;
            cl_uint numPlatforms;
            status = clGetPlatformIDs(0, NULL, &numPlatforms);

            if (status != CL_SUCCESS)
			{
				throw std::runtime_error("Error: Getting Platforms\n");
			}

            return numPlatforms;
        }

        std::vector<cl_platform_id> get_all_platforms()
        {
            cl_int status = 0;
			cl_uint numPlatforms = platform_num();

			if (numPlatforms > 0)
			{
				std::vector<cl_platform_id> platforms(numPlatforms);
				status = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
				if (status != CL_SUCCESS)
				{
					throw std::runtime_error("Error: Getting Platform Ids.(clGetPlatformIDs)\n");
				}
                return platforms;
			}

            return {};
        }

        cl_platform_id select_first_platform(const std::vector<cl_platform_id>& platforms)
        {
            cl_int status = 0;
            cl_platform_id platform = nullptr;
            for (unsigned int i = 0; i < platforms.size(); ++i)
            {
                char pbuff[100];
                status = clGetPlatformInfo(
                    platforms[i],
                    CL_PLATFORM_VENDOR,
                    sizeof(pbuff),
                    pbuff,
                    NULL);
                platform = platforms[i];
                break;
            }
        }

        cl_context_properties* get_context_property(cl_platform_id platform)
        {
            cl_context_properties cps[3] = {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)platform,
                0
            };

            cl_context_properties *cprops = (nullptr == platform) ? nullptr : cps;
            return cps;
        }
    };
}

#endif