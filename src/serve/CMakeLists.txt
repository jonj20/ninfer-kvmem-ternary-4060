# The portable launcher reads the same model catalog before it loads a CUDA backend.
add_library(ninfer_model_catalog STATIC model_catalog.cpp)
ninfer_internal_includes(ninfer_model_catalog)

add_library(ninfer_serve STATIC
  http_transport.cpp
  console_log.cpp
  request_validation.cpp
  structured_output.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  text_completion.cpp
  rerank.cpp
  systemone.cpp
  decisions_http.cpp
  llamacpp_http.cpp
  openai_common.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  branch_memo.cpp
  request_events.cpp
  console_stats.cpp
  operational_log.cpp
  request_log.cpp
  http_server.cpp
  mcp_proxy.cpp
  load_report.cpp
  serve_metrics.cpp
  openai_responses_http.cpp
  serve_options.cpp
  stop_control.cpp
  model_registry.cpp
)
ninfer_internal_includes(ninfer_serve)
target_link_libraries(ninfer_serve PUBLIC kvmem PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(ninfer_serve
  PUBLIC ninfer_engine ninfer_model_catalog Threads::Threads ninfer::json ninfer::httplib
  PRIVATE ninfer_media_acquire ninfer_product_logging ninfer_product_prompt_input
          ${NINFER_CUDART_TARGET})

# The WebUI is compiled in only from local directories, never downloaded at configure time, so
# offline and hermetic builds are unaffected. scripts/fetch-webui.sh (scripts/fetch-webui.bat on
# Windows) unpacks llama.cpp's released WebUI into such a directory; NINFER_WEBUI_DIR names it.
# The monitoring view is ours and ships with every build: it reads the server's own /stats and
# /api/monitor/snapshot and is served from the same port, under /kvmem/.
set(NINFER_WEBUI_DIR "" CACHE PATH
    "Directory of built WebUI files (with index.html) to serve from ninfer-serve; empty serves the monitoring view alone")
set(NINFER_WEBUI_MONITOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/webui/kvmem")
add_executable(ninfer_webui_embed webui/embed.cpp)
file(GLOB_RECURSE NINFER_WEBUI_MONITOR_FILES CONFIGURE_DEPENDS "${NINFER_WEBUI_MONITOR_DIR}/*")
set(NINFER_WEBUI_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/webui_assets.cpp")
set(NINFER_WEBUI_EMBED_SPECS "${NINFER_WEBUI_MONITOR_DIR}=kvmem")
if(NINFER_WEBUI_DIR)
  if(NOT EXISTS "${NINFER_WEBUI_DIR}/index.html" AND NOT EXISTS "${NINFER_WEBUI_DIR}/index.html.gz")
    message(FATAL_ERROR "NINFER_WEBUI_DIR has no index.html: ${NINFER_WEBUI_DIR}")
  endif()
  file(GLOB_RECURSE NINFER_WEBUI_FILES CONFIGURE_DEPENDS "${NINFER_WEBUI_DIR}/*")
  list(INSERT NINFER_WEBUI_EMBED_SPECS 0 "${NINFER_WEBUI_DIR}")
endif()
add_custom_command(
  OUTPUT "${NINFER_WEBUI_SOURCE}"
  COMMAND ninfer_webui_embed "${NINFER_WEBUI_SOURCE}" ${NINFER_WEBUI_EMBED_SPECS}
  DEPENDS ninfer_webui_embed ${NINFER_WEBUI_FILES} ${NINFER_WEBUI_MONITOR_FILES}
  COMMENT "Embedding the WebUI and the monitoring view"
  VERBATIM)
target_sources(ninfer_serve PRIVATE "${NINFER_WEBUI_SOURCE}")
