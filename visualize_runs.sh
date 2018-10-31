#!/usr/bin/env bash
# Creates plots for all runs inside a root directory. The names of the subdirectories may be arbitrary, but they
# must conform to the expected structure. The CSV-exported training- and validation-losses from tensorboard must be present
cd ./src/
root_dir=$1
subdirs=$(find ${root_dir} -maxdepth 1 -mindepth 1 -type d -printf '%f\n')
for subdir in ${subdirs}
do
    subdir=${root_dir}/${subdir}
    echo "visualizing $subdir ..."
    python3 ./visualize_learning_curve.py ${subdir} --silent
    echo "...done"
done

