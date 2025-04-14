find_package(OpenCV REQUIRED COMPONENTS core imgproc)
if (MSVC)
  set(DEPENDENCIES_TO_COPY
    "${__location_release}"
  )
else()
  set(DEPENDENCIES_TO_COPY
      ${OpenCV_DIR}/../../libopencv_core.so.4.5d
      ${OpenCV_DIR}/../../libopencv_imgproc.so.4.5d
      ${TBB_DIR}/../../libtbb.so.2
  )
endif()
