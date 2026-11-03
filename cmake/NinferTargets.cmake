# Internal headers are private to each compile owner. External headers come from
# the dependency targets (or a component-local include for bundled C sources).
function(ninfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)
endfunction()

# Device code is compressed in every fatbin: the INT8-family small-T attention alone instantiates
# several hundred kernels per query width (route tiers x KV codings x batch and mask forms), and
# uncompressed the ops archive passed the 2 GiB that a linked image can address with 32-bit
# relocations. Compressed for size it is about 3.5x smaller; the driver expands a module when it
# loads.
set(NINFER_CUDA_FATBIN_OPTIONS "-Xfatbin=-compress-all")
# The default (speed) mode barely shrinks these kernels; the size mode, from CUDA 12.8, does.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  list(APPEND NINFER_CUDA_FATBIN_OPTIONS "-compress-mode=size")
endif()

if(NINFER_CUDA_PRUNE_PTX)
  get_filename_component(_ninfer_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  find_program(NINFER_NVPRUNE nvprune HINTS "${_ninfer_cuda_bin}" REQUIRED)
endif()

function(ninfer_prune_cuda_ptx target)
  get_target_property(_ninfer_library_type ${target} TYPE)
  if(NINFER_CUDA_PRUNE_PTX AND _ninfer_library_type STREQUAL "STATIC_LIBRARY")
    # Archive-only processing preserves compiler-cache entries and all selected GPU targets.
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
        "-DNVPRUNE=${NINFER_NVPRUNE}"
        "-DARCHIVE=$<TARGET_FILE:${target}>"
        "-DARCHITECTURES=$<TARGET_PROPERTY:${target},CUDA_ARCHITECTURES>"
        -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PruneCudaPtx.cmake"
      VERBATIM)
  endif()
endfunction()

# Whole-program device code: every device symbol is defined and used in one translation unit (no
# extern __device__ or __constant__, no cross-unit device calls), and relocatable device code made
# ptxas assume external linkage, lower computed-lane shuffles to out-of-line calls and give up
# pipelining loads across loop back edges.
function(ninfer_cuda_archive target)
  if(WIN32)
    set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    "$<$<COMPILE_LANGUAGE:CUDA>:${NINFER_CUDA_FATBIN_OPTIONS}>")
  ninfer_prune_cuda_ptx(${target})
endfunction()

function(ninfer_cuda_non_rdc_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  if(WIN32)
    set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    "$<$<COMPILE_LANGUAGE:CUDA>:${NINFER_CUDA_FATBIN_OPTIONS}>")
  ninfer_prune_cuda_ptx(${target})
endfunction()

# Links a static archive whole into a Windows DLL and exports the `expected` explicit
# instantiations of the function template `symbol` it defines, and nothing else of it. Automatic
# exports (WINDOWS_EXPORT_ALL_SYMBOLS) read only the DLL's own objects; CMake merges this
# module-definition file into them. The archive's sources stay as every other build compiles them.
# The archive's own link step writes the file. As the output of a custom command of the DLL, it
# would make Ninja compile the archive before any object of a target that links the DLL.
function(ninfer_export_archive_instantiations dll archive symbol expected)
  get_filename_component(_ninfer_linker_dir "${CMAKE_LINKER}" DIRECTORY)
  find_program(NINFER_DUMPBIN dumpbin HINTS "${_ninfer_linker_dir}" REQUIRED)
  set(_ninfer_definitions "${CMAKE_CURRENT_BINARY_DIR}/${archive}_exports.def")
  add_custom_command(TARGET ${archive} POST_BUILD
    COMMAND ${CMAKE_COMMAND}
      "-DDUMPBIN=${NINFER_DUMPBIN}"
      "-DARCHIVE=$<TARGET_FILE:${archive}>"
      "-DSYMBOL=${symbol}"
      "-DEXPECTED=${expected}"
      "-DOUTPUT=${_ninfer_definitions}"
      -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ArchiveExports.cmake"
    BYPRODUCTS "${_ninfer_definitions}"
    VERBATIM)
  target_sources(${dll} PRIVATE "${_ninfer_definitions}")
  target_link_libraries(${dll} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${archive}>")
endfunction()

# Runtime DLL staging for the apps and tests.
#
# ninfer / ninfer-serve (and fully-linked test binaries) statically import the
# FFmpeg + libcurl runtime DLLs (avutil/avcodec/avformat/swscale/swresample/libcurl
# plus their own deps, e.g. z.dll). The vcpkg IMPORTED_LINK_DEPENDENT_LIBRARIES hook
# does not reliably stage them into the output tree, and the Windows loader will not
# search the vcpkg bin dir when the exe is launched directly -- it only searches the
# exe's own directory, system dirs, and PATH. An unresolved import aborts the process
# before main (0xC0000135). Stage the full closure next to each binary so it runs
# standalone.

# Collect the FFMPEG + libcurl runtime DLLs from the vcpkg bin tree into <out_var>.
function(ninfer_runtime_dll_list out_var)
  set(_list "")
  if(WIN32 AND NINFER_WINDOWS_KERNEL_DLL AND TARGET ninfer_kernels)
    list(APPEND _list "$<TARGET_FILE:ninfer_kernels>")
  endif()
  if(WIN32 AND DEFINED FFMPEG_VCPKG_TREE)
    foreach(_ninfer_dll_glob
      "avutil-*.dll" "avcodec-*.dll" "avformat-*.dll" "swscale-*.dll"
      "swresample-*.dll" "libcurl*.dll" "z.dll")
      file(GLOB _ninfer_dlls "${FFMPEG_VCPKG_TREE}/bin/${_ninfer_dll_glob}")
      list(APPEND _list ${_ninfer_dlls})
    endforeach()
    list(REMOVE_DUPLICATES _list)
  endif()
  set(${out_var} "${_list}" PARENT_SCOPE)
endfunction()

function(ninfer_stage_runtime_dlls target)
  ninfer_runtime_dll_list(_ninfer_rt_dlls)
  if(NOT _ninfer_rt_dlls)
    if(WIN32 AND DEFINED FFMPEG_VCPKG_TREE)
      message(WARNING
        "ninfer: no runtime DLLs found under ${FFMPEG_VCPKG_TREE}/bin; "
        "skipping staging for ${target}")
    endif()
    return()
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${_ninfer_rt_dlls} $<TARGET_FILE_DIR:${target}>
    COMMENT "Staging runtime DLLs (ffmpeg/curl) next to ${target}")
endfunction()

# Stage the shared FFMPEG/curl closure once next to the test executables. All test
# exes share a single per-config runtime directory, so one copy serves every one of
# them (anchored on any always-built test target for the directory).
function(ninfer_stage_test_runtime_dlls anchor_target)
  if(NOT WIN32)
    return()
  endif()
  ninfer_runtime_dll_list(_ninfer_test_rt_dlls)
  if(NOT _ninfer_test_rt_dlls)
    return()
  endif()
  add_custom_target(ninfer_stage_test_runtime_dlls ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_ninfer_test_rt_dlls}
            "$<TARGET_FILE_DIR:${anchor_target}>"
    COMMENT "Staging runtime DLLs (ffmpeg/curl) next to test executables")
endfunction()
