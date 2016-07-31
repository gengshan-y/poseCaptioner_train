import os
import h5py
import time
import csv
import logging


''' Stores the split result in format {'train':, 'val':, 'test':, 'info':}
    info contains split ratio and clip number '''
def randSplit(splitRat, inPath, outPath, clipRange):
    import numpy
    import json
    from random import shuffle

    """ get all lines from file """
    with open(inPath) as f:
        clipList = f.readlines()
    clipList = clipList[clipRange[0]:clipRange[1]]

    clipNum = len(clipList)
    logging.info(str(clipNum) + ' titles loaded')

    """ create a random ordering of the lines """
    idxList = range(0, clipNum)
    shuffle(idxList)

    splitCumRat = [splitRat[0], 1 - splitRat[2]]
    splitRange = [int(it) for it in numpy.multiply(clipNum, splitCumRat)]

    """ Set splits """
    out = {}
    out['train'] = [clipList[it][:-1] for it in idxList[: splitRange[0]]]
    out['val'] = [clipList[it][:-1] for it in idxList[splitRange[0]: splitRange[1]]]
    out['test'] = [clipList[it][:-1] for it in idxList[splitRange[1]:]]
    out['info'] = splitRat + [clipNum]

    ''' dump results '''
    json.dump(out, open(outPath, 'w'))


h5InPath = '/data/gengshan/pose/all.h5'  # hg_img_all.h5  # all.h5
svOutPath = '/data/gengshan/pose_s2vt/splits/'

""" Transfer data from .h5/json to csv/tsv files """
def prepareData(split):
    """ prepare h5 file reader """
    h5InFile = h5py.File(h5InPath, 'r')  
    
    """ prepare file writer """
    csvOutPath = svOutPath + 'dataCsv_' + split['name'] + '.txt'
    tsvOutPath = svOutPath + 'dataTsv_' + split['name'] + '.txt'
    csvFile = open(csvOutPath, 'a')
    tsvFile = open(tsvOutPath, 'a')
    csvWriter =csv.writer(csvFile, delimiter = ',', lineterminator='\n')
    tsvWriter =csv.writer(tsvFile, delimiter = '\t', lineterminator='\n')
    logging.info(csvOutPath + ' opened for writing')
    logging.info(tsvOutPath + ' opened for writing')
     
    beg = time.time()
    for it, clipTitle in enumerate(split['titleList']):
        logging.info(clipTitle + ": ")
        """ Get captions from .txt """
        if not os.path.exists('/data2/gengshan/clip/' + clipTitle.rsplit('_', 1)[0] + '/' + clipTitle + '.txt'):
            logging.info('/data2/gengshan/clip/' + clipTitle.rsplit('_', 1)[0] + '/' + clipTitle + '.txt' + "doesn't exist")
            continue
        with open('/data2/gengshan/clip/' + clipTitle.rsplit('_', 1)[0] + '/' + clipTitle + '.txt') as tempCapFIle:
            tempCap = ''
            for tempSentence in tempCapFIle:
                tempCap += tempSentence.replace("\n", " ")  # remove "newline"
        
        """ Get pose features of a clip """
        if not clipTitle in h5InFile.keys():
            logging.info(clipTitle + ' not in the dataset')
            continue
        frameFeatArray = {}
        for itt,frameFeat in enumerate(h5InFile[clipTitle].keys()):
            frameFeatArray[frameFeat] = h5InFile[clipTitle][frameFeat][:].flatten()  # get the feat and flatten it 
        frameFeatArray = sorted(frameFeatArray.items())  # sort by key

        """ Write csv file and tsv file """ 
        for (key, value) in frameFeatArray:
            csvWriter.writerow([key] + value.tolist())

        tsvWriter.writerow([clipTitle] + [tempCap])
        logging.info([clipTitle] + [tempCap])

        '''if it % 100 == 0:
            logging.info('combined ' + str(it) + ' ...\n')
            logging.info(str(time.time() - beg) + 's lapsed\n')
            beg = time.time()
        '''

    """ Close files """
    tsvFile.close()
    csvFile.close()
    h5InFile.close()


"""  """
def prepareSplitData(taskID, batchSize, splitTitles, splitName):
    split = {}
    if taskID * batchSize >= len(splitTitles):
        logging.info('Lowerbound exceeded.')
        return False
    
    elif (taskID + 1) * batchSize > len(splitTitles):
        logging.info('Upperbound exceeded. Cut in the mid.')
        split['name'] = splitName
        split['titleList'] = splitTitles[taskID * batchSize: ]
        
    else:
        logging.info('Good, move on.')
        split['name'] = splitName
        split['titleList'] = splitTitles[taskID * batchSize: (taskID + 1) * batchSize]
    prepareData(split)
    return True
