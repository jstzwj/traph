#ifndef TRAPH_CORE_TYPE_H_
#define TRAPH_CORE_TYPE_H_

#include <variant>
#include <optional>
#include <cstdint>

namespace traph
{
    using f32 = float;
    using f64 = double;
    using i8 = std::int8_t;
    using i16 = std::int16_t;
    using i32 = std::int32_t;
    using i64 = std::int64_t;
    using u8 = std::uint8_t;
    using u16 = std::uint16_t;
    using u32 = std::uint32_t;
    using u64 = std::uint64_t;
    using grad_type = f32;
    using idx_type = i32;
    using size_type = i32;
    using device_id = i32;

    enum layout_type
    {
        row_major,
        column_major
    };

    enum PlatformType
    {
        CPU,
        CUDA,
        OPENCL,
        VULKAN,
        OPENGL
    };

    enum DataType
    {
        BYTE,
        CHAR,
        SHORT,
        INT,
        LONG,
        FLOAT,
        DOUBLE
    };

    class ScalarType
    {
    private:
        std::variant<u8, i8, i16, i32, i64, f32, f64> _scalar;
        DataType _dtype;
    public:
		ScalarType(u8 v)
			:_scalar(v) {}

		ScalarType(i8 v)
			:_scalar(v) {}

		ScalarType(i16 v)
			:_scalar(v) {}

		ScalarType(i32 v)
			:_scalar(v) {}

		ScalarType(i64 v)
			:_scalar(v) {}

		ScalarType(f32 v)
			:_scalar(v) {}

		ScalarType(f64 v)
			:_scalar(v) {}

        DataType dtype() const
        {
            return _dtype;
        }

        std::optional<u8> get_byte()
        {
            if(_dtype == DataType::BYTE)
                return std::get<u8>(_scalar);
            else
                return std::nullopt;
        }

        std::optional<i8> get_char()
        {
            if(_dtype == DataType::CHAR)
                return std::get<i8>(_scalar);
            else
                return std::nullopt;
        }

        std::optional<i16> get_short()
        {
            if(_dtype == DataType::SHORT)
                return std::get<i16>(_scalar);
            else
                return std::nullopt;
        }

        std::optional<i32> get_int()
        {
            if(_dtype == DataType::INT)
                return std::get<i32>(_scalar);
            else
                return std::nullopt;
        }

        std::optional<i64> get_long()
        {
            if(_dtype == DataType::LONG)
                return std::get<i64>(_scalar);
            else
                return std::nullopt;
        }

        std::optional<f32> get_float()
        {
            if(_dtype == DataType::FLOAT)
                return std::get<f32>(_scalar);
            else
                return std::nullopt;
        }

        std::optional<f64> get_double()
        {
            if(_dtype == DataType::DOUBLE)
                return std::get<f64>(_scalar);
            else
                return std::nullopt;
        }
    };
}

#endif