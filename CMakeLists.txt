cmake_minimum_required(VERSION 3.20)
project(test_pytorch)

#Opciones de compilación
set(CMAKE_CXX_STANDARD 17)


set(programs  predictionbatch_RGBNIR_CPU predictionbatch_RGBNIR_CUDA save_inputtensor convert_tensor)

set(CAFFE2_USE_CUDNN 0) # Para que libtorch use cuDNN

if(NOT WIN32)
  set (CUDA_HOME /usr/local/cuda)
  SET(CMAKE_INSTALL_BINDIR $ENV{HOME}/bin)
  set (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /opt/libtorch) # En Linux para que findPackage encuentre torch
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  FIND_PACKAGE(X11 REQUIRED)
	
	FIND_PACKAGE(Threads REQUIRED)
  
	include_directories(${X11_INCLUDE_DIR})
	link_directories(${X11_LIB_DIRS})
	SET(SYSTEM_LIBS ${X11_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()


if( WIN32)
  SET(CMAKE_INSTALL_BINDIR $ENV{USERPROFILE}/bin)
  set (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} C:/libtorch/share/cmake/Torch/)
  set (CMAKE_MODULE_PATH ${CMAKE_MODEULE_PATH} C:/libtorch/share/cmake/Torch/)
  set (CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} C:/opencv/build/x64/vc16/lib)
  set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc")
  set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/")
  
  #Esto deberia sacarse del findPackage de libtorch
  link_directories(C:/libtorch/lib/)
  link_directories("C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64")
  link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64")
  include_directories(C:/libtorch/include)  
endif()


message("Install BIN DIR " ${CMAKE_INSTALL_BINDIR})

# Buscar OpenCV y Torch
# En Windows: La variable de entorno OPENCV_DIR debe estar bien configurada
#           : Para Torch?
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

message("TORCH_LIBRARIES " ${TORCH_LIBRARIES})
message("TORCH_INCLUDE_DIRS " ${TORCH_INCLUDE_DIRS})
message("TORCH_CXX_FLAGS " ${TORCH_CXX_FLAGS})

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_compile_options(${TORCH_CXX_FLAGS})
include_directories(${TORCH_INCLUDE_DIRS})

# Es posible que la lista de directorios de include se pueda obtener de TORCH_INCLUDE_DIRS	
if(NOT WIN32)
  include_directories(/opt/libtorch/include)
else()
  include_directories(C:/libtorch/include) 
endif()


foreach (prog    ${programs})
    add_executable(${prog} ${prog}.cpp)
    target_link_libraries(${prog}     
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${SYSTEM_LIBS}
)


endforeach(prog)


install(TARGETS  ${programs}
DESTINATION ${CMAKE_INSTALL_BINDIR})
