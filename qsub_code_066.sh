
#!/bin/bash
#$ -pe threaded 1
#$ -N evo_ldc066
#$ -cwd
#$ -j y
#$ -S /bin/bash

#echo "NHOSTS=$NHOSTS, NSLOTS=$NSLOTS"

# Paths and includes from the user (the packages installed by pip using the --user option ...)
#PATH_PYTHON37_INCLUDE=/home/taleslf/.conda/envs/tese_sloth/bin/python3.7
#export PATH=$PATH:$PATH_PYTHON37_INCLUDE

#LD_LIBRARY_PATH=/home/taleslf/lib64:/home/taleslf/GCC-7.1.0/lib64:/usr/local/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH

# Run the program-
my_pids=( 0 )
/share/apps/anaconda3/bin/python3.7 /home/goliatt/longitudinal_dispersion_coefficient/ldc_evoml_regression_v0p3.py -r 066 > /home/goliatt/longitudinal_dispersion_coefficient/ldc_evoml___066.out 2>&1 &
my_pids[0]=$!
wait ${my_pids[@]}
    
