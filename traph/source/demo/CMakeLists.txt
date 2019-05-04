SET(LIB_NAME demo)
SET(LIB_OUTNAME traph-${LIB_NAME})

SET(HEADER_PATH ${TRAPH_PATH_HEADER}/${LIB_NAME})
SET(SOURCE_PATH ${TRAPH_PATH_SOURCE}/${LIB_NAME})

SET(DEMO_LIST
	${SOURCE_PATH}/main.cpp
)

add_executable(${LIB_OUTNAME} ${DEMO_LIST})
target_link_libraries(${LIB_OUTNAME} traph-core)
target_link_libraries(${LIB_OUTNAME} traph-tensor)
target_link_libraries(${LIB_OUTNAME} traph-nn)


