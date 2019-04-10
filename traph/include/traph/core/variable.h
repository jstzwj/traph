#ifndef TRAPH_CORE_VARIABLE_H_
#define TRAPH_CORE_VARIABLE_H_


namespace traph
{
    template<class T>
    class VariableBase
    {
    public:
        virtual platform_type platform() = 0;

        virtual device_id device() = 0;
        
        virtual void reshape(const DimVector& dims) = 0;

        virtual void resize(const DimVector& dims) = 0;

		virtual idx_type offset() const = 0;

		virtual layout_type layout() const = 0;

		virtual DimVector size() const = 0;

		virtual const T* data() const = 0;
		virtual T* data() = 0;

		virtual DimVector strides() const = 0;
    };
}

#endif