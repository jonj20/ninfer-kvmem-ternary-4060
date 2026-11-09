target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/append/kv_cache_append.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/append/launch.cu"
)
if(NINFER_KV_ATTN_NVFP4)
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/append/nvfp4_launch.cu")
endif()
if(NINFER_KV_ATTN_K8V4)
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/append/k8v4_launch.cu")
endif()
if(NOT NINFER_KV_ATTN_NVFP4 OR NOT NINFER_KV_ATTN_K8V4)
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/append/disabled_stubs.cpp")
endif()

# The switch definitions reach only the append units that read them.
set_source_files_properties(
  "${CMAKE_CURRENT_LIST_DIR}/append/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/append/disabled_stubs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/append/nvfp4_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/append/k8v4_launch.cu"
  PROPERTIES COMPILE_DEFINITIONS "${NINFER_KV_ATTN_COMPILE_DEFINITIONS}")
