%module traph_tensor
%{
    #include<traph/core/type.h>
    #include<traph/core/tensor.h>
    #include<traph/core/index.h>
    using namespace traph;
%}

typedef float f32;
typedef double f64;
typedef std::int8_t i8;
typedef std::int16_t i16;
typedef std::int32_t i32;
typedef std::int64_t i64;
typedef std::uint8_t u8;
typedef std::uint16_t u16;
typedef std::uint32_t u32;
typedef std::uint64_t u64;
typedef i32 idx_type;
typedef i32 size_type;

%typemap(in) idx_type {
  $1 = PyInt_AsLong($input);
}

%typemap(out) idx_type {
  $result = PyInt_FromLong($1);
}

%typemap(in) size_type {
  $1 = PyInt_AsLong($input);
}

%typemap(out) size_type {
  $result = PyInt_FromLong($1);
}

class DimVector
{
private:
    std::unique_ptr<idx_type[]> data;
    idx_type stack_data[DIMVECTOR_SMALL_VECTOR_OPTIMIZATION];
    idx_type dim_num;
public:
    DimVector();
    DimVector(idx_type size);
    DimVector(const DimVector& other);
    DimVector(DimVector&& other);
    DimVector& operator=(const DimVector& other) noexcept;
    DimVector& operator=(DimVector&& other) noexcept;
    void push_back(idx_type idx);
    void resize(idx_type size);
    idx_type size() const;
    idx_type& operator[](idx_type dim);
    idx_type operator[](idx_type dim) const;
};

template<class T>
class TensorStorage
{
private:
    std::unique_ptr<T[]> data;
    idx_type len;
public:
    TensorStorage();
    TensorStorage(const TensorStorage& other);
    TensorStorage(TensorStorage&& other);
    TensorStorage& operator=(const TensorStorage& other);
    TensorStorage& operator=(TensorStorage&& other);

    idx_type size() const;
    size_type element_size() const;
    void resize_(idx_type size);
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

    void reshape(const DimVector& dims);
    T& index(const DimVector& dims);
};

%template(ByteTensor) Tensor<u8>;
%template(CharTensor) Tensor<i8>;
%template(ShortTensor) Tensor<i16>;
%template(IntTensor) Tensor<i32>;
%template(LongTensor) Tensor<i64>;
%template(FloatTensor) Tensor<f32>;
%template(DoubleTensor) Tensor<f64>;

%template(ByteStorage) TensorStorage<u8>;
%template(CharStorage) TensorStorage<i8>;
%template(ShortStorage) TensorStorage<i16>;
%template(IntStorage) TensorStorage<i32>;
%template(LongStorage) TensorStorage<i64>;
%template(FloatStorage) TensorStorage<f32>;
%template(DoubleStorage) TensorStorage<f64>;
