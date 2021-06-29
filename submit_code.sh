files=`pwd`/*qsub_code*sh 


for i in $files;
do
echo $i
done

for i in $files;
do
qsub  $i
sleep 2s
done
