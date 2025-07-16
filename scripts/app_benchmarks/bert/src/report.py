import os,argparse

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
            if metric == 'Time for shard':
                val=float(tmp[tmp.index('seconds')-1])
            elif 'progress' in metric:
                for word in tmp:
                    if 'loss' in word:
                        val=float(word.strip('\n ,').split('=')[1])
            elif 'SamplesPerSec' in metric:
                for word in tmp:
                    if 'Samples' in word:
                        val=float(word.strip('\n ,').split('=')[1])
    return val

if __name__ == '__main__':
    time_for_shard_1    = process(buf,'Time for shard')
    loss                = process(buf,'bing_bert_progress')
    samples_per_sec     = process(buf,'SamplesPerSec')
    
    print('Time for Shard 1: %f\nSamplesPerSec: %f\nLoss: %f\n' %(time_for_shard_1,samples_per_sec,loss))
      


