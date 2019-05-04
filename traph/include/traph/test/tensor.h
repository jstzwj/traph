#ifndef TRAPH_TEST_TENSOR_H_
#define TRAPH_TEST_TENSOR_H_

#include <catch2/catch.hpp>
#include <traph/core/index.h>

TEST_CASE( "DimVector test", "[DimVector]" )
{
    traph::DimVector dim;
    SECTION("smaller than stack optimization size")
    {
        dim.resize(3);

        REQUIRE(dim.size() == 3);
    }

    SECTION("larger than stack optimization size")
    {
        dim.resize(10);

        REQUIRE(dim.size() == 10);
    }

    SECTION("zero size")
    {
        dim.resize(0);

        REQUIRE(dim.size() == 0);
    }
}

#endif