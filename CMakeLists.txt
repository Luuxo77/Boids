cmake_minimum_required(VERSION 3.21.1)

project(Boids LANGUAGES CUDA C CXX)

find_package(OpenGL REQUIRED)

set(SOURCE_FILES
    src/helper.h
    src/Boids.h
    src/Parameters.cuh
    src/kernels.cuh
    src/kernels.cu
    src/glad.c
    src/Boids.cpp
    src/main.cpp
    src/fragmentShader.txt
    src/vertexShader.txt
    src/geometryShader.txt
)
set(IMGUI_SOURCE_FILES
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_demo.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/imgui.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_impl_glfw.cpp
    src/imgui/imgui_impl_opengl3.cpp
)

include_directories(include)
link_directories(lib)

add_executable(Boids ${SOURCE_FILES} ${IMGUI_SOURCE_FILES})

target_include_directories(Boids PRIVATE)
target_link_libraries(Boids PRIVATE glfw3)

add_custom_command(
        TARGET Boids POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROJECT_SOURCE_DIR}/src/geometryShader.txt
                ${CMAKE_CURRENT_BINARY_DIR}/geometryShader.txt)

add_custom_command(
	TARGET Boids POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy
        	${PROJECT_SOURCE_DIR}/src/geometryShader.txt
        	$<TARGET_FILE_DIR:Boids>)   
     
add_custom_command(
        TARGET Boids POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROJECT_SOURCE_DIR}/src/fragmentShader.txt
                ${CMAKE_CURRENT_BINARY_DIR}/fragmentShader.txt)

add_custom_command(
	TARGET Boids POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy
        	${PROJECT_SOURCE_DIR}/src/fragmentShader.txt
        	$<TARGET_FILE_DIR:Boids>)      
add_custom_command(
        TARGET Boids POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROJECT_SOURCE_DIR}/src/vertexShader.txt
                ${CMAKE_CURRENT_BINARY_DIR}/vertexShader.txt)

add_custom_command(
	TARGET Boids POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy
        	${PROJECT_SOURCE_DIR}/src/vertexShader.txt
        	$<TARGET_FILE_DIR:Boids>)              
