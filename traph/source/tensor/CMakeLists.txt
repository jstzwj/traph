SET(LIB_NAME tensor)
SET(LIB_OUTNAME traph-${LIB_NAME})

SET(HEADER_PATH ${TRAPH_PATH_HEADER}/${LIB_NAME})
SET(SOURCE_PATH ${TRAPH_PATH_SOURCE}/${LIB_NAME})

SET(TENSOR_LIST
	${HEADER_PATH}/float_tensor.h
	${SOURCE_PATH}/float_tensor.cpp
	${HEADER_PATH}/double_tensor.h
	${SOURCE_PATH}/double_tensor.cpp
	${HEADER_PATH}/char_tensor.h
	${SOURCE_PATH}/char_tensor.cpp
	${HEADER_PATH}/byte_tensor.h
	${SOURCE_PATH}/byte_tensor.cpp
	${HEADER_PATH}/short_tensor.h
	${SOURCE_PATH}/short_tensor.cpp
	${HEADER_PATH}/int_tensor.h
	${SOURCE_PATH}/int_tensor.cpp
	${HEADER_PATH}/long_tensor.h
	${SOURCE_PATH}/long_tensor.cpp
	${HEADER_PATH}/tensor.h
	${SOURCE_PATH}/tensor.cpp
	${HEADER_PATH}/arithmetic.h
	${SOURCE_PATH}/arithmetic.cpp
)

ADD_LIBRARY(${LIB_OUTNAME} ${TENSOR_LIST})

IF(Boost_FOUND)
	target_link_libraries(${LIB_OUTNAME} ${Boost_LIBRARIES})
ENDIF()

IF(TRAPH_ACCELERATE EQUAL 0)
	# link eigen?
ELSEIF(TRAPH_ACCELERATE EQUAL 1)
	if (BLAS_FOUND)
		target_link_libraries(${LIB_OUTNAME} ${BLAS_LIBRARIES})
	endif()
ELSEIF(TRAPH_ACCELERATE EQUAL 2)
	if (BLAS_FOUND)
		target_link_libraries(${LIB_OUTNAME} ${BLAS_LIBRARIES})
	endif()
ELSE()
	MESSAGE(FATAL_ERROR "Unsupported build platform: " ${OCTOON_BUILD_PLATFORM})
ENDIF()



