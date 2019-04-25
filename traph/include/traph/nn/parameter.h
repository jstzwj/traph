#ifndef TRAPH_NN_PARAMETER_H_
#define TRAPH_NN_PARAMETER_H_

#include <traph/nn/variable.h>

namespace traph
{
    template<typename T>
    class Parameter:public Variable<T>
    {
    public:
        Parameter();
        Parameter(std::shared_ptr<TensorBase<T>> data);
        Parameter(const DimVector& dim);
        Parameter(std::initializer_list<idx_type> l);

		Parameter(const Parameter& other) = delete;
		Parameter(Parameter&& other) = delete;
        Parameter& operator= (const Parameter& other) = delete;
        Parameter& operator= (Parameter&& other) = delete;

        ~Parameter();
    };

    template<class T>
    using ParameterPtr = std::shared_ptr<Parameter<T>>;
    template<class T>
    using ParameterRef = Parameter<T>&;
    template<class T>
    using ParameterConstRef = const Parameter<T>&;

    // only support these type template, so methods defined in source is ok...
    template class Parameter<u8>;
    template class Parameter<i8>;
    template class Parameter<i16>;
    template class Parameter<i32>;
    template class Parameter<i64>;
    template class Parameter<f32>;
    template class Parameter<f64>;

    using DoubleParameter = Parameter<f64>;
    using FloatParameter = Parameter<f32>;
    using LongParameter = Parameter<i64>;
    using IntParameter = Parameter<i32>;
    using ShortParameter = Parameter<i16>;
    using CharParameter = Parameter<i8>;
    using ByteParameter = Parameter<u8>;


    // definition
	template<typename T>
	Parameter<T>::Parameter()
		:Variable<T>()
	{

	}

	template<typename T>
	Parameter<T>::Parameter(std::shared_ptr<TensorBase<T>> data)
		:Variable<T>(data)
	{
	}

	template<typename T>
	Parameter<T>::Parameter(const DimVector& dim)
		:Variable<T>(dim)
	{
	}

	template<typename T>
	Parameter<T>::Parameter(std::initializer_list<idx_type> l)
		:Variable<T>(l)
	{
	}

	template<typename T>
	Parameter<T>::~Parameter()
	{
	}
}

#endif