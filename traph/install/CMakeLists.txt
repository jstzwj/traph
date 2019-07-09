INCLUDE(ExternalProject)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
	MESSAGE(STATUS "current platform: Linux ")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    ExternalProject_Add(tbb
        URL               https://github.com/intel/tbb/releases/download/2019_U8/tbb2019_20190605oss_win.zip
        URL_HASH          MD5=82A1A72BF97F3F35C1C57036A92951E4
        PREFIX ${TRAPH_PATH_INSTALL}/source/tbb
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
    )
    SET(TBB_INCLUDE_DIRS ${TRAPH_PATH_INSTALL}/source/tbb/src/tbb/tbb2019_20190605oss/include)
    SET(TBB_LIBRARIES ${TRAPH_PATH_INSTALL}/source/tbb/src/tbb/tbb2019_20190605oss/lib)
ELSE ()
	MESSAGE(STATUS "other platform tbb is not supported: ${CMAKE_SYSTEM_NAME}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")


ExternalProject_Add(
    xtl
    GIT_REPOSITORY https://github.com/QuantStack/xtl.git
    GIT_TAG 0d1f896ba90664248279baaea0059699cff5ab9b
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    PREFIX ${TRAPH_PATH_INSTALL}/source/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TRAPH_PATH_INSTALL} -DBUILD_SHARED_LIBS=OFF
)

ExternalProject_Add(
    xsimd
    GIT_REPOSITORY https://github.com/QuantStack/xsimd.git
    GIT_TAG 898402de2db78fa82c7f0df016b64ffa51dbd212
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    PREFIX ${TRAPH_PATH_INSTALL}/source/xsimd
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TRAPH_PATH_INSTALL} -DBUILD_SHARED_LIBS=OFF
)

ExternalProject_Add(
    xtensor
    GIT_REPOSITORY https://github.com/QuantStack/xtensor.git
    GIT_TAG d1513e465a44fef5a2b86aece05ec2157d1bd343
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    PREFIX ${TRAPH_PATH_INSTALL}/source/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TRAPH_PATH_INSTALL} -DBUILD_SHARED_LIBS=OFF
        -DTBB_LIBRARIES_RELEASE=ON -DTBB_INCLUDE_DIRS=${TBB_INCLUDE_DIRS} -DTBB_LIBRARIES=${TBB_LIBRARIES}
        -DXTENSOR_USE_XSIMD=ON -DXTENSOR_ENABLE_ASSERT=ON -DXTENSOR_CHECK_DIMENSION=ON -DXTENSOR_USE_TBB=ON
)

add_dependencies(xtensor xsimd xtl tbb)

INCLUDE_DIRECTORIES(${TRAPH_PATH_INSTALL}/include)