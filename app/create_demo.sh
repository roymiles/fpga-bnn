#!/bin/bash

# AUTHOR: ROY MILES (student)
# The software, hardware, and hardware with shared object lib files are all built with a common interface using app.cpp
# Therefore, can easily test all 3 configurations under the same environment with the same app.cpp API
# Just call the appropriate output executable "program_sw", "program_hwnolib" etc

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <mode>" >&2
  echo "where <mode> = hw_lib hw_nolib sw hw_so" >&2
  exit 1
fi

MODE=$1

TINY_CNN_PATH=/opt/python3.6/lib/python3.6/site-packages/bnn/src/xilinx-tiny-cnn
OPENCV_PATH=/opt/opencv/include
OPENCV_LIBS="-lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videoio -lopencv_videostab"
BNN_ROOT=/opt/python3.6/lib/python3.6/site-packages/bnn/src
BNN_SO=/opt/python3.6/lib/python3.6/site-packages/bnn/libraries

SW_PYNQ=$BNN_ROOT/network/cnv-pynq/sw

# ap_int.h
#VIVADOHLS_INCLUDE_PATH=/home/xilinx/vivado_include
VIVADOHLS_INCLUDE_PATH=/home/xilinx/open_cv2/hls

# foldedmv-offload.h rawhls.cpp etc
HOST_INC_PATH=$BNN_ROOT/library/host

# Contains xlnk drivers, donutdriver etc
DRIVER_INC_PATH=$BNN_ROOT/library/driver

# bnn-library.h convlayer.h etc - used by software implementation
HLS_INC_PATH=$BNN_ROOT/library/hls

#SRCS_MAIN=ExtractROI.cpp
#SRCS_MAIN=test.cpp
SRCS_MAIN=app.cpp

# config.h (a copy of config.h is in the current directory)
# CNV_PYNQ_INC=$BNN_ROOT/network/cnv-pynq/hw

SRCS_OPENCV_UTILS=/home/xilinx/open_cv2/opencv_utils.cpp

#USE_LIB=1
if [ $MODE == 'hw_lib' ]
then
	echo "Using library..."
	SRCS_FOLDEDMV="/home/xilinx/open_cv2/lib/lib-foldedmv-offload.cpp"
	SRCS_RAWHLS="/home/xilinx/open_cv2/lib/lib-rawhls-offload.cpp"
	SRCS_SW_PYNQ="/home/xilinx/open_cv2/lib/lib-main_python.cpp"
	
	SRCS_ALL="$SRCS_MAIN $SRCS_FOLDEDMV $SRCS_RAWHLS $SRCS_SW_PYNQ $SRCS_OPENCV_UTILS"
	
	EXTRA="-L/home/xilinx/open_cv2/ -lkernelbnn"
	
	OUTPUT_NAME="program_lib"
elif [ $MODE == 'hw_nolib' ]
then
	# Using xlnk donut driver (no raw hls)
	echo "Using xlnk drivers"
	SRCS_FOLDEDMV="/home/xilinx/open_cv2/foldedmv-offload.cpp"
	SRCS_XLNK=$BNN_ROOT/library/driver/platform-xlnk.cpp
	SRCS_SW_PYNQ="/home/xilinx/open_cv2/main_python.cpp"
	#SRCS_KMEANS_KDTREE="/home/xilinx/open_cv2/build_kdTree.cpp"
	#SRCS_KMEANS_UTILS="/home/xilinx/open_cv2/kmeans/filtering_algorithm_util.cpp"
	#SRCS_KMEANS_STACK="/home/xilinx/open_cv2/kmeans/stack.cpp"
	
	#SRCS_ALL="$SRCS_MAIN $SRCS_XLNK $SRCS_FOLDEDMV $SRCS_SW_PYNQ $SRCS_OPENCV_UTILS $SRCS_KMEANS_KDTREE $SRCS_KMEANS_UTILS $SRCS_KMEANS_STACK"
	SRCS_ALL="$SRCS_MAIN $SRCS_XLNK $SRCS_FOLDEDMV $SRCS_SW_PYNQ $SRCS_OPENCV_UTILS"
	
	EXTRA=""
	
	OUTPUT_NAME="program_nolib"
elif [ $MODE == 'sw' ]
then
	echo "Software implementation"
	
	SRCS_FOLDEDMV="/home/xilinx/open_cv2/sw/sw-foldedmv-offload.cpp"
	SRCS_RAWHLS="/home/xilinx/open_cv2/sw/sw-rawhls-offload.cpp"
	SRCS_SW_PYNQ="/home/xilinx/open_cv2/sw/sw-main_python.cpp"
	SRCS_TOP="/home/xilinx/open_cv2/sw/top.cpp" # Definition of BlackBoxJam (uses config.h in CNV_PYNQ_INC)
	
	SRCS_ALL="$SRCS_MAIN $SRCS_FOLDEDMV $SRCS_RAWHLS $SRCS_SW_PYNQ $SRCS_OPENCV_UTILS $SRCS_TOP"
	
	EXTRA=""
	
	OUTPUT_NAME="program_sw"
elif [ $MODE == 'hw_so' ]
then
	echo "Hardware shared object library (provided by pynq)"
	echo "not yet implemented.."
	exit 1
	SRCS_FOLDEDMV="/home/xilinx/open_cv2/sw/sw-foldedmv-offload.cpp"
	SRCS_SW_PYNQ="/home/xilinx/open_cv2/sw/sw-main_python.cpp"
	
	SRCS_ALL="$SRCS_MAIN $SRCS_FOLDEDMV $SRCS_SW_PYNQ $SRCS_OPENCV_UTILS $SRCS_TOP"
	
	EXTRA="-L$BNN_SO -lpython_hw-cnv-pynq"
	
	OUTPUT_NAME="program_hw_sw"
else
	echo "Unknown mode parameter"
	exit 1
fi

LOC_SW_INC="/home/xilinx/open_cv2/sw/"
LOC_LIB_INC="/home/xilinx/open_cv2/lib/"

# So the build can find the opencv pkg config file (contains linker references to opencv libraries)
export PKG_CONFIG_PATH=/opt/opencv/lib/pkgconfig
# Add -g flag if want to debug
# -O3 generates standard compliant programs, -Ofast does not (but better optimisation)
# -o0 makes it compile faster
# g++ vs clang++
CMD="g++ -std=c++17 -pthread -O3 -v -z muldefs -DXILINX -DOFFLOAD --std=gnu++11 $SRCS_ALL -o $OUTPUT_NAME -I/home/xilinx/open_cv2/hls/hls -I$LOC_SW_INC -I$LOC_LIB_INC -I$DRIVER_INC_PATH -I$VIVADOHLS_INCLUDE_PATH -I$TINY_CNN_PATH -I$HLS_INC_PATH `pkg-config --cflags --libs opencv` -lpthread -lsds_lib $EXTRA"
# -L /home/xilinx/host/bnn_lib_tests/lib_bnn -l kernelbnn
# -DNEON -mfpu=neon -funsafe-math-optimizations -ftree-vectorize -mvectorize-with-neon-quad -ftree-vectorizer-verbose=2
echo $CMD
$CMD # Run command
