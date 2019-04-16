%module traph_tensor

%begin %{
#ifdef _MSC_VER
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#endif
%}

%{
    #include <string>
    #include<traph/core/type.h>
    #include<traph/core/index.h>
    #include<traph/tensor/tensor.h>
    #include<traph/tensor/tensor_storage.h>
    
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

%typemap(in) std::string {
  PyObject* bytes = PyUnicode_AsUTF8String($input);
  $1 = PyBytes_AsString(bytes);
}

%typemap(out) std::string {
  $result = PyUnicode_FromString($1.data());
}

class DimVector
{
public:
    DimVector();
    DimVector(idx_type size);
    DimVector(const DimVector& other);
    DimVector(DimVector&& other);
    DimVector& operator=(const DimVector& other) noexcept;
    DimVector& operator=(DimVector&& other) noexcept;
    void erase(idx_type idx);
    void push_back(idx_type idx);
    void resize(idx_type size);
    idx_type size() const;
    idx_type flat_size() const;
    idx_type& operator[](idx_type dim);
    idx_type operator[](idx_type dim) const;
};

template<typename T>
class TensorStorage
{
public:
    TensorStorage();
    TensorStorage(const TensorStorage& other);
    TensorStorage(TensorStorage&& other);

    TensorStorage& operator=(const TensorStorage& other);
    TensorStorage& operator=(TensorStorage&& other);

    virtual std::shared_ptr<StorageBase<T>> clone() const override;
    virtual T* data_ptr() override;
    virtual const T* data_ptr() const override;
    virtual idx_type size() const override;
    virtual size_type element_size() const override;

    virtual void resize_(idx_type size) override;

    // fill
    virtual void fill_(T v) override;
};

%template(ByteStorage) TensorStorage<u8>;
%template(CharStorage) TensorStorage<i8>;
%template(ShortStorage) TensorStorage<i16>;
%template(IntStorage) TensorStorage<i32>;
%template(LongStorage) TensorStorage<i64>;
%template(FloatStorage) TensorStorage<f32>;
%template(DoubleStorage) TensorStorage<f64>;

template<class T>
class Tensor
{
public:
  Tensor();
  explicit Tensor(const DimVector& dimensions);
  explicit Tensor(const DimVector& dimensions, layout_type order);
  explicit Tensor(const DimVector& dimensions, const DimVector& strides);
  explicit Tensor(const DimVector& dimensions, const DimVector& strides, layout_type order);
  Tensor(const T& t);

  Tensor(const Tensor& other) = delete;
  Tensor(Tensor&& other) = delete;
  Tensor& operator= (const Tensor& other) = delete;
  Tensor& operator= (Tensor&& other) = delete;

  virtual void add_(TensorInterfacePtr other) override;
  virtual void apply_(std::function<T(T)> f) override;
  virtual TensorInterfacePtr clone() const override;
  virtual void cos_() override;
  virtual std::shared_ptr<TensorBase<f32>> create_grad() override;
  virtual T* data_ptr() override;
  virtual const T* data_ptr() const override;
  virtual device_id device() override;
  virtual void fill_(T value) override;
  virtual T item() const override;
  virtual idx_type offset() const override;
  virtual layout_type order() const override;
  virtual platform_type platform() override;
  virtual T reduce_(std::function<T(T,T)> f) const override;
  virtual TensorInterfacePtr reduce_dim(idx_type dim, std::function<T(T,T)> f) const override;
  virtual void reshape_(const DimVector& dims) override;
  virtual void resize_(const DimVector& dims) override;
  virtual void sin_() override;
  virtual DimVector size() const override;
  virtual idx_type size(idx_type i) const override;
  virtual std::shared_ptr<StorageBase<T>> storage() const override;
  virtual DimVector stride() const override;
  virtual idx_type stride(idx_type i) const override;
  virtual TensorInterfacePtr sum() const override;
  virtual std::string to_string() const override;
};

%template(ByteTensor) Tensor<u8>;
%template(CharTensor) Tensor<i8>;
%template(ShortTensor) Tensor<i16>;
%template(IntTensor) Tensor<i32>;
%template(LongTensor) Tensor<i64>;
%template(FloatTensor) Tensor<f32>;
%template(DoubleTensor) Tensor<f64>;


