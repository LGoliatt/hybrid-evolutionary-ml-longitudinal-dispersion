
import sys
v=sys.argv
if len(v)>1:
    r1,r2= int(v[1]),int(v[2])
else:
    r1,r2=1,30

import os
hostname=os.uname()[1]


os.system('mkdir output')

scriptname='ldc_evoml_regression_v0p3.py'
s='#\n'
for i in range(r1-1,r2):
  j="{:02d}".format(i+1)
  s+='python3.5 '+scriptname+' -r  '+str(j)+' > ./output/'+scriptname.split('.')[0]+'__'+hostname+'_'+str(j)+'.out'+' &'
  s+='\n'
  
print(s)

shname=scriptname.split('.')[0]+'__'+hostname+'.sh'

with open(shname, 'w') as f:
    f.write(s)


