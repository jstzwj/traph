#ifndef TRAPH_INDEX_H
#define TRAPH_INDEX_H

#include <cstdint>
#include <cstring>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <utility>
#include <traph/core/type.h>

#define DIMVECTOR_SMALL_VECTOR_OPTIMIZATION 4

namespace traph
{
    class DimVector
    {
    private:
        std::unique_ptr<idx_type[]> data;
        idx_type stack_data[DIMVECTOR_SMALL_VECTOR_OPTIMIZATION];
        idx_type dim_num;
    public:
        DimVector()
            :data(nullptr), dim_num(0)
        {
        }

        DimVector(idx_type size)
        {
            if(size < 0)
                return;
            if(size > DIMVECTOR_SMALL_VECTOR_OPTIMIZATION)
            {
                data = std::make_unique<idx_type[]>(size);
            }
            
            dim_num = size;
        }

        DimVector(const DimVector& other)
        {
            if(other.dim_num > DIMVECTOR_SMALL_VECTOR_OPTIMIZATION)
            {
                data = std::make_unique<idx_type[]>(other.dim_num);
                std::memcpy(data.get(), other.data.get(), other.dim_num * sizeof(idx_type));
            }
            else
            {
                std::memcpy(stack_data, other.stack_data, other.dim_num * sizeof(idx_type));
            }
            dim_num = other.dim_num;
        }

        DimVector(DimVector&& other)
        {
            if(other.dim_num > DIMVECTOR_SMALL_VECTOR_OPTIMIZATION)
            {
                data = std::move(other.data);
            }
            else
            {
                std::memcpy(stack_data, other.stack_data, other.dim_num * sizeof(idx_type));
            }
            dim_num = other.dim_num;
        }

        DimVector& operator=(const DimVector& other) noexcept
        {
            if(other.dim_num > DIMVECTOR_SMALL_VECTOR_OPTIMIZATION)
            {
                data = std::make_unique<idx_type[]>(other.dim_num);
                std::memcpy(data.get(), other.data.get(), other.dim_num * sizeof(idx_type));
            }
            else
            {
                std::memcpy(stack_data, other.stack_data, other.dim_num * sizeof(idx_type));
            }
            dim_num = other.dim_num;
            return *this;
        }

        DimVector& operator=(DimVector&& other) noexcept
        {
            if(other.dim_num > DIMVECTOR_SMALL_VECTOR_OPTIMIZATION)
            {
                data = std::move(other.data);
            }
            else
            {
                std::memcpy(stack_data, other.stack_data, other.dim_num * sizeof(idx_type));
            }
            dim_num = other.dim_num;
            return *this;
        }

        void push_back(idx_type idx)
        {
            resize(size() + 1);
            this->operator[](size() - 1) = idx;
        }

        void resize(idx_type size)
        {
            if(size < 0 || size == dim_num)
                return;
            if(size > DIMVECTOR_SMALL_VECTOR_OPTIMIZATION)
            {
                if(dim_num > DIMVECTOR_SMALL_VECTOR_OPTIMIZATION)
                {
                    idx_type move_size = (size > dim_num ? dim_num: size);
                    std::unique_ptr<idx_type[]> temp(new idx_type[size]);
                    std::memcpy(temp.get(), data.get(), move_size * sizeof(idx_type));
                    data = std::move(temp);
                }
                else
                {
                    data = std::unique_ptr<idx_type[]>(new idx_type[size]);
                    std::memcpy(data.get(), stack_data, dim_num * sizeof(idx_type));
                }
            }
            else
            {
                if(dim_num > DIMVECTOR_SMALL_VECTOR_OPTIMIZATION)
                {
                    data = std::unique_ptr<idx_type[]>(nullptr);
                    if (size != 0)
                        std::memcpy(stack_data, data.get(), size * sizeof(idx_type));
                }
            }
            dim_num = size;
        }

        idx_type size() const { return this->dim_num; }

        idx_type& operator[](idx_type dim)
        {
            if(dim<0 || dim >= dim_num)
                throw std::runtime_error("index out of dim vector size");
            
            if(dim_num > DIMVECTOR_SMALL_VECTOR_OPTIMIZATION)
                return data[dim];
            else
                return stack_data[dim];
        }

        idx_type operator[](idx_type dim) const
        {
            if(dim<0 || dim >= dim_num)
                throw std::runtime_error("index out of dim vector size");
            
            if(dim_num > DIMVECTOR_SMALL_VECTOR_OPTIMIZATION)
                return data[dim];
            else
                return stack_data[dim];
        }
    };
}

#endif