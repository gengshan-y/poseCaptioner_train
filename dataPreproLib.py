import os
import h5py
import time
import csv
import logging

""" Transfer data from .h5/json to csv/tsv files """
def prepareData(split):
    """ prepare h5 file reader """
    h5InFile = h5py.File('/data/gengshan/pose/all.h5', 'r')
    
    """ prepare file writer """
    csvFile = open('/data/gengshan/pose_s2vt/splits/' + 'dataCsv_' + split['name'] + '.txt', 'a')
    tsvFile = open('/data/gengshan/pose_s2vt/splits/' + 'dataTsv_' + split['name'] + '.txt', 'a')
    csvWriter =csv.writer(csvFile, delimiter = ',', lineterminator='\n')
    tsvWriter =csv.writer(tsvFile, delimiter = '\t', lineterminator='\n')
     
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

        """ Write csv file and tsv file """ 
        for key, value in frameFeatArray.items():
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
