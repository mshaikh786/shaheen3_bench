import os,argparse
import re

parser = argparse.ArgumentParser(description='Report generator for BERT benchmark.')
parser.add_argument('--file', type=str,
                    help='Filename with absolute path to process')
args = parser.parse_args()


fh = open(args.file, 'r', encoding='utf-8', errors='ignore')

buf = fh.readlines()

fh.close()

def process(buf,metric):
    val=0.0
    for line in buf:
        if metric in line:
            tmp = line.strip().split()
            if metric == 'Total Execution Time':
                val=float(tmp[tmp.index('seconds')-1])
            elif 'EPOCHS' in metric:
                val = [0,0,0]
                tmp = re.split(': |, ',line.strip())
                for word in tmp:
                    if 'Epoch time' in word:
                        val[0]=float(tmp[tmp.index('Epoch time')+1])
                    elif 'Batch Time' in word:
                        val[1]=float(tmp[tmp.index('Batch Time')+1])
                    elif 'Data Time' in word:
                        val[2]=float(tmp[tmp.index('Data Time')+1])
            elif 'Average-mAP' in metric:
                tmp = re.split(':|, ',line.strip())
                for word in tmp:
                    val=float(tmp[tmp.index('Average-mAP')+1])
    return val

if __name__ == '__main__':
    execution_time    = process(buf,'Total Execution Time')
    avg_mAP           = process(buf, 'Average-mAP')
    epochs_metrics    = process(buf, 'EPOCHS')
    
    print('Total Execution Time: ', execution_time)
    print('Average_mAP:', avg_mAP)
     
    print('Epoch time:',epochs_metrics[0])
    print('Batch time:',epochs_metrics[1])
    print('Data time:',epochs_metrics[2])
    
      


