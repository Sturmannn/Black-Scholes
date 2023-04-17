#!/bin/bash
echo -n > ./Zameri.txt
for i in {1..5}
do
  ./build/Kokkos_version/Kokkos_Version 
done

./measurements/Get_Measurements

echo "Сalculation has been completed"
