cmake_minimum_required(VERSION 3.25.0) #Проверка версии CMake.

set(PROJECT_NAME Black-Scholes_OPM) #Создаем обычную локальную переменную с именем проекта
project(${PROJECT_NAME}) # Название решения

#Переменные с именем проектов
set(serial Serial_Version)
set(openMP Parallel_openMP_version)
set(kokkos Kokkos_Version)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE) #Создается переменная CMAKE_CONFIGURATION_TYPES со значением "Debug;Release" и меткой/разделом Configs, переменная сохраниется в специальном файле доступном из всех файлов,
if(NOT CMAKE_BUILD_TYPE) # проверка имеющихся у компилятора вариантов сборки
  set(CMAKE_BUILD_TYPE Release) #Если нет то устанавливаем Release
endif()

#Добавляем директорию с под-CMakeList-ом, и выполняем под-CMakeList
add_subdirectory(Serial_version)
add_subdirectory(Parallel_openMP_version)
add_subdirectory(Kokkos_version)
add_subdirectory(kokkosLib)
