%module traph_tensor
%{
    #include<traph/core/type.h>
    #include<traph/core/tensor.h>
    using namespace traph;
%}

template<class T>
class TensorStorage
{
public:
    std::unique_ptr<T[]> data;
    idx_type len;

    TensorStorage();
    void resize(idx_type size);
    void resize(const DimVector& dimensions);
};

template<class T>
class Tensor
{
private:
    std::unique_ptr<TensorStorage<T>> rep;
    DimVector dimensions;
    idx_type offset;
    DimVector strides;
    layout_type order;
public:
    Tensor();
    Tensor(const DimVector& dimensions);
    Tensor(const DimVector& dimensions, layout_type order);
    Tensor(const DimVector& dimensions, const DimVector& strides);
    Tensor(const DimVector& dimensions, const DimVector& strides, layout_type order);
};

%template(tensor_f32) Tensor<f32>;
%template(tensor_f64) Tensor<f64>;