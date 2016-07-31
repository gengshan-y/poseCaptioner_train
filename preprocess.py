from dataPreproLib import *
import json
import numpy as np
import os
import time
import argparse

''' logger '''
import sys
sys.path.insert(0, '/home/gengshan/workJul/mergeCap')
import logInit 
logpath = 'log/log_'
logger = logInit.makeLogger(logpath)

''' Parsing arguments '''
parser = argparse.ArgumentParser()
parser.add_argument("-beg", "--beg", nargs='?', default=0, type=int,
                    help='begining index of clips')
parser.add_argument("-end", "--end", nargs='?', default=None, type=int,
                    help='ending index of clips')
parser.add_argument("-bat", "--batchSize", nargs='?', default=10, type=int,
                    help='number of samples written to files in one batch')

args = parser.parse_args()
logging.info(args)

""" split info """
splitRat = [0.6, 0.2, 0.2]  # [train, valid, test]
clipTitlePath = '/home/gengshan/public_html/data/clipTitles.txt'
splitInfoPath = '/data/gengshan/pose_s2vt/splits/splitInfo.json'
clipRange = (args.beg, args.end)

""" writing sv file info """
batchSize = args.batchSize
outName = ['train', 'val', 'test']  # output dataset name

''' Load split title and calculate batch size '''
randSplit(splitRat, clipTitlePath, splitInfoPath, clipRange)
splitInfo = json.load(open(splitInfoPath, 'r'))  
splitData = [splitInfo['train'], splitInfo['val'], splitInfo['test']]
splitBatchSize = [int(it) for it in \
                              np.multiply(batchSize, splitInfo['info'][:3])]

''' delete before appending data '''
for itt in [0, 1, 2]:
    os.system('rm /data/gengshan/pose_s2vt/splits/*' + outName[itt] + '.txt')

''' write data in append mode '''
beg = time.time()
res = True
it = 0
while res:
    for itt in [0, 1, 2]:  # train, val, test
        res = prepareSplitData(it, splitBatchSize[itt], splitData[itt], outName[itt])
    if it % 100 == 0:
        logging.info(str(it * batchSize) + ' samples written')
        logging.info('lapsed ' + str(time.time() - beg))
    it += 1

''' print info '''
logging.info(str(it - 1) + ' batches loaded')
logging.info('lapsed ' + str(time.time() - beg))
logger.handlers[0].close()
