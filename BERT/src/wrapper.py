import os
import subprocess as sb
import sys
ntasks=int(sys.argv[1])
print("ntasks: ",ntasks)
hosts = sb.run(['mpirun','-np', '%s' %ntasks, '--map-by','node:PE=1','/bin/hostname'],stdout=sb.PIPE,stderr=sb.PIPE)
if hosts.stderr.decode('utf-8') is not "":
      print('stderr: ',hosts.stderr.decode('utf-8'))

hosts=hosts.stdout.decode('utf-8').strip().split('\n')
f = open('hostfile', 'w')
num_gpus = int(os.getenv('NUM_GPUS'))
for host in hosts:
  f.write("%s slots=%d\n" %(host,num_gpus))

f.close()


   



