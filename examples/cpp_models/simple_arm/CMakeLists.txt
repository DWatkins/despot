cmake_minimum_required(VERSION 2.8.3)

add_executable("${PROJECT_NAME}_simple_arm"
  src/simple_arm.cpp
  src/main.cpp
)
target_link_libraries("${PROJECT_NAME}_simple_arm"
  "${PROJECT_NAME}"
)
install(TARGETS "${PROJECT_NAME}_simple_arm"
  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
)
