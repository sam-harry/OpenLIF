#!/bin/sh

for dir in still_water solitary_wave_breaking undular_bore_patching
do
  if [[ -d ${dir} ]]
  then
    cd ${dir}
    echo starting ${dir}
    python3 -m cProfile -o profile.out run.py
    echo finished ${dir}
    cd ..
  fi
done
