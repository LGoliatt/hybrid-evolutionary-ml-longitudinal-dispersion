
qsub_file = open("submit_code_all.sh", "w")


for i in range(100):
    n = str(i+1).zfill(3)    
    
    s='''
#!/bin/bash
#$ -pe threaded 1
#$ -N evo_ldc'''+str(n)+'''
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
/share/apps/anaconda3/bin/python3.7 /home/goliatt/longitudinal_dispersion_coefficient/ldc_evoml_regression_v0p3.py -r '''+n+''' > /home/goliatt/longitudinal_dispersion_coefficient/ldc_evoml___'''+n+'''.out 2>&1 &
my_pids[0]=$!
wait ${my_pids[@]}
    '''
    

    #text_file = open("qsub_code_"+n+".sh", "w")
    
    with open("qsub_code_"+n+".sh", "w") as text_file:
        print(s, file=text_file)
    
    qsub_file.write("qsub "+"qsub_code_"+n+".sh & \n")


qsub_file.close()






