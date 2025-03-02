cmake_minimum_required(VERSION 2.8.3)
project(okvis_ros)

find_package(catkin REQUIRED  
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  image_transport
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  image_geometry
  ## Added by Sharmin
  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  message_generation

  imagenex831l
  #aquacore #for aqua Depth topic

  ## Added by Hunter
  std_srvs
  nav_msgs
)

# generate dynamic reconfigure stuff
generate_dynamic_reconfigure_options(cfg/Camera.cfg)

# Sharmin: for SvinHealth msg
add_message_files(
  FILES
  SvinHealth.msg
)
add_service_files(
  FILES
  OdometryTrigger.srv
)
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
)

catkin_package(
    DEPENDS
    CATKIN_DEPENDS geometry_msgs
                   tf 
                   tf2
                   roscpp
                   tf2_ros
                   tf2_geometry_msgs
                   tf2_sensor_msgs
                   sensor_msgs
                   message_runtime
                   std_srvs
                   nav_msgs
                   cv_bridge
                   message_filters
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)

# we really want to use Release here
set(CMAKE_BUILD_TYPE Release)

message(STATUS "Building with ${CMAKE_BUILD_TYPE}")

# The configure step fails at catkin_package() if this directory
# doesn't exist yet. Create it at configure time!
file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/include)

# let's pull in the okvis plain cmake project
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(okvis)


# libvisensor
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/okvis/cmake")
find_package(VISensor)

if(VISENSORDRIVER_FOUND)
  add_definitions(-DHAVE_LIBVISENSOR)
  message(STATUS "Found libvisensor in ROS interface.")
  include_directories(${VISensorDriver_INCLUDE_DIR})
else()
  message(WARNING "libvisensor not found")
endif()

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/include
)

# compile the ROS dependent stuff into a library

#removed -g flag which produces debubugging information 
#-g is not needed for release builds

add_definitions(-fPIC -march=native -O3 -Wextra -Winit-self -Woverloaded-virtual
                -Wnon-virtual-dtor -Wsign-promo -Wno-long-long -std=c++17)

add_library(${PROJECT_NAME} STATIC src/Subscriber.cpp
                               src/Publisher.cpp
                               src/RosParametersReader.cpp
                               include/okvis/Subscriber.hpp
                               include/okvis/Publisher.hpp
                               include/okvis/RosParametersReader.hpp)

add_dependencies(${PROJECT_NAME} okvis_multisensor_processing)

# make sure configure headers are built before any node using them
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_generate_messages_cpp)  # Sharmin

target_link_libraries(${PROJECT_NAME}  
  ${catkin_LIBRARIES}
  okvis_util
  okvis_kinematics
  okvis_time
  okvis_cv 
  okvis_common
  okvis_ceres
  okvis_timing
  okvis_matcher
  okvis_frontend 
  okvis_multisensor_processing
  pthread
)

if(VISENSORDRIVER_FOUND)
  target_link_libraries(${PROJECT_NAME} ${VISensorDriver_LIBRARY})
endif()

include_directories(
  okvis/okvis_util/include
  okvis/okvis_kinematics/include
  okvis/okvis_time/include
  okvis/okvis_cv/include
  okvis/okvis_common/include
  okvis/okvis_ceres/include
  okvis/okvis_timing/include
  okvis/okvis_matcher/include
  okvis/okvis_frontend/include 
  okvis/okvis_multisensor_processing/include
)

# now the actual applications
add_executable(okvis_node src/okvis_node.cpp )
target_link_libraries(okvis_node ${PROJECT_NAME} )
#add_executable(okvis_node_synchronous src/okvis_node_synchronous.cpp)
#target_link_libraries(okvis_node_synchronous ${PROJECT_NAME} )
add_executable(dataset_convertor src/dataset_convertor.cpp)
target_link_libraries(dataset_convertor ${PROJECT_NAME} )

add_executable(stereo_sync src/stereo_sync.cpp)
target_link_libraries(stereo_sync ${catkin_LIBRARIES} ${PROJECT_NAME})
