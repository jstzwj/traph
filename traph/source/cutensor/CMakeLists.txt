SET(LIB_NAME cutensor)
SET(LIB_OUTNAME traph-${LIB_NAME})

SET(HEADER_PATH ${TRAPH_PATH_HEADER}/${LIB_NAME})
SET(SOURCE_PATH ${TRAPH_PATH_SOURCE}/${LIB_NAME})

# add_subdirectory(cuda)

SET(CUTENSOR_LIST
	${HEADER_PATH}/cutensor.h
	${SOURCE_PATH}/cutensor.cpp
)

ADD_LIBRARY(${LIB_OUTNAME} ${CUTENSOR_LIST})
target_link_libraries(${LIB_OUTNAME} traph-cutensor-cuda)

IF(Boost_FOUND)
	target_link_libraries(${LIB_OUTNAME} ${Boost_LIBRARIES})
ENDIF()


IF(TRAPH_ACCELERATE EQUAL 0)
	# do something
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



