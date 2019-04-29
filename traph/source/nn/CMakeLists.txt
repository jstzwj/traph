SET(LIB_NAME nn)
SET(LIB_OUTNAME traph-${LIB_NAME})

SET(HEADER_PATH ${TRAPH_PATH_HEADER}/${LIB_NAME})
SET(SOURCE_PATH ${TRAPH_PATH_SOURCE}/${LIB_NAME})

SET(NN_LIST
	${HEADER_PATH}/autograd.h
	${HEADER_PATH}/variable.h
	${SOURCE_PATH}/variable.cpp
	${HEADER_PATH}/graph.h
	${HEADER_PATH}/executor.h
	${SOURCE_PATH}/executor.cpp
	${HEADER_PATH}/function.h
	${HEADER_PATH}/operation.h
	${SOURCE_PATH}/operation.cpp
	${HEADER_PATH}/optim.h
)

ADD_LIBRARY(${LIB_OUTNAME} ${NN_LIST})

IF(Boost_FOUND)
	target_link_libraries(${LIB_OUTNAME} ${Boost_LIBRARIES})
ENDIF()

IF(OpenCL_FOUND)
	target_link_libraries(${LIB_OUTNAME} ${OpenCL_LIBRARIES})
ENDIF()

IF(TRAPH_ACCELERATE EQUAL 0)
	# using eigen
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



