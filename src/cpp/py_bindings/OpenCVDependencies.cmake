find_package(OpenCV REQUIRED COMPONENTS core imgproc)
if (MSVC)
  set(DEPENDENCIES_TO_COPY
    "${__location_release}"
  )
else()
  find_package(PkgConfig REQUIRED)
  find_package(TBB "2021.5.0" EXACT REQUIRED)
  pkg_check_modules(TBB REQUIRED tbb)

  set(DEPENDENCIES_TO_COPY
      ${OpenCV_DIR}/../../libopencv_core.so.4.5d
      ${OpenCV_DIR}/../../libopencv_imgproc.so.4.5d
      ${pkgcfg_lib_TBB_tbb}.2 # ubuntu system package uses tbb.so.2
  )
endif()

set(PYTHON_PACKAGE_DIR ${VISION_API_SOURCE_DIR})
foreach(lib ${DEPENDENCIES_TO_COPY})
    add_custom_command(
        TARGET _vision_api POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${lib}
            ${PYTHON_PACKAGE_DIR}
        COMMENT "Copying ${lib} to ${PYTHON_PACKAGE_DIR}"
    )
endforeach()
