from random import shuffle
import json
import numpy


""" set input info """
splitRat = [0.6, 0.2, 0.2]  # [train, valid, test]
inPath = '/home/gengshan/public_html/data/clipTitles.txt'
outPath = '/data/gengshan/pose_s2vt/splits/splitInfo.json'

#########################################################
""" get all lines from file """
with open(inPath) as f:
    clipList = f.readlines()
clipList = clipList[:1000]  # get minibatch

clipNum = len(clipList)
print(str(clipNum) + ' titles loaded')

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
