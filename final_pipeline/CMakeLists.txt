cmake_minimum_required (VERSION 3.13)
project (final_pipeline)

#change to your raspicam build directory
list(APPEND CMAKE_MODULE_PATH "/home/pi/Documents/EEE190/raspicam-0.1.6/build")

find_package(raspicam REQUIRED)
find_package(OpenCV)

IF  ( OpenCV_FOUND AND raspicam_CV_FOUND)

#change to the path of your RaspberryPi TFLite C++ static library
set(RPI_TFLITE_LIBPATH "/home/pi/Documents/EEE190/tensorflow/tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a")

#change to the path of your TFlite repo local directory
set(TFLITE_INCLUDEPATH_1 "/home/pi/Documents/EEE190/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include")
set(TFLITE_INCLUDEPATH_2 "/home/pi/Documents/EEE190/tensorflow/")

MESSAGE(STATUS "COMPILING PIPELINE")

add_executable (liveSSD_pipeline liveSSD_pipeline.cpp ${TFLITE_INCLUDEPATH_2}/tensorflow/lite/interpreter.h ${TFLITE_INCLUDEPATH_2}/tensorflow/lite/kernels/register.h ${TFLITE_INCLUDEPATH_2}/tensorflow/lite/model.h ${TFLITE_INCLUDEPATH_2}/tensorflow/lite/optional_debug_tools.h)
target_include_directories(liveSSD_pipeline PUBLIC ${TFLITE_INCLUDEPATH_1} ${TFLITE_INCLUDEPATH_2})

#The below line doesn't work so use legacy command set(CMAKE_CXX_FLAGS) for now 
#target_compile_options(liveSSD_pipeline PUBLIC -Wall -Wextra -pedantic -pthread)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -pthread")
target_link_libraries (liveSSD_pipeline ${raspicam_CV_LIBS} ${RPI_TFLITE_LIBPATH})
ELSE()
MESSAGE(FATAL_ERROR "OPENCV NOT FOUND IN YOUR SYSTEM")
ENDIF()
