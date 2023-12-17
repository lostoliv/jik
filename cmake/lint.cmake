set(CMAKE_BUILD_DIR .)
set(CMAKE_SOURCE_DIR ..)

# We download the latest cpplint version
file(DOWNLOAD https://raw.githubusercontent.com/google/styleguide/gh-pages/cpplint/cpplint.py ${CMAKE_BUILD_DIR}/cpplint.py)
set(LINT_COMMAND python3 ${CMAKE_BUILD_DIR}/cpplint.py)

# Directories and files to run cpplint on
set(SRC_FILE_EXTENSIONS h cc)
set(SRC_DIRS core recurrent sandbox)

# Find all files of interest
foreach(ext ${SRC_FILE_EXTENSIONS})
  foreach(dir ${SRC_DIRS})
    file(GLOB_RECURSE FOUND_FILES ${CMAKE_SOURCE_DIR}/${dir}/*.${ext})
    set(SRC_FILES ${SRC_FILES} ${FOUND_FILES})
  endforeach()
endforeach()

# Run cpplint
foreach(file ${SRC_FILES})
  message(STATUS "Checking " ${file})
  execute_process(COMMAND ${LINT_COMMAND} ${file})
endforeach()
