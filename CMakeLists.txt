cmake_minimum_required(VERSION 2.8) #Проверка версии CMake.

set(PROJECT_NAME Black-Scholes_OPM) #Создаем обычную локальную переменную с именем проекта
project(${PROJECT_NAME}) # Название проекта

#Переменные с именем приложения
set(serial Serial_Version)
set(openMP Parallel_openMP_version)
set(DPC Parallel_DPC++_version)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE) #Создается переменная CMAKE_CONFIGURATION_TYPES со значением "Debug;Release" и меткой/разделом Configs, переменная сохраниется в специальном файле доступном из всех файлов,
if(NOT CMAKE_BUILD_TYPE) # проверка имеющихся у компилятора вариантов сборки
  set(CMAKE_BUILD_TYPE Release) #Если нет то устанавливаем Release
endif()

#Добавляем директорию с под-CMakeList-ом, и выполняем под-CMakeList
add_subdirectory(Serial_version)
add_subdirectory(Parallel_openMP_version)
add_subdirectory(Parallel_DPC++_version)
