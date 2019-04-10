SET(LIB_NAME cltensor)
SET(LIB_OUTNAME traph-${LIB_NAME})

SET(HEADER_PATH ${TRAPH_PATH_HEADER}/${LIB_NAME})
SET(SOURCE_PATH ${TRAPH_PATH_SOURCE}/${LIB_NAME})

SET(CORE_LIST
	${HEADER_PATH}/cltensor.h
	${SOURCE_PATH}/cltensor.cpp
	${HEADER_PATH}/device.h
	${SOURCE_PATH}/device.cpp
)

ADD_LIBRARY(${LIB_OUTNAME} ${CORE_LIST})

IF(Boost_FOUND)
	target_link_libraries(${LIB_OUTNAME} ${Boost_LIBRARIES})
ENDIF()

IF(OpenCL_FOUND)
	target_link_libraries(${LIB_OUTNAME} ${OpenCL_LIBRARIES})
ENDIF()

IF(TRAPH_ACCELERATE EQUAL 1)
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



