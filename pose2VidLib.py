import numpy as np
from PIL import Image, ImageDraw

''' 
    Pose Reference:
    local pairRef = {
        {1,2},      {2,3},      {3,7},
        {4,5},      {4,7},      {5,6},
        {7,9},      {9,10},
        {14,9},     {11,12},    {12,13},
        {13,9},     {14,15},    {15,16}
    }

    local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
                       'Pelv','Thrx','Neck','Head',
                       'RWri','RElb','RSho','LSho','LElb','LWri'}
'''

pairRef = [(8, 9 ), (9,10), (14,9), (11, 12), (12, 13), (13, 9), (14, 15), (15, 16)]  # remove everyting about hip, knee and ankle
pairRef = [(it[0] - 1, it[1] -1) for it in pairRef]


''' Get pose data with format 
    {'clipTitle': [string], 'imgPose': [16*2 float with 256 resolution]}
    and clipTitles '''
def getStructuredData(dataPath):
    structuredData = []
    
    ''' read data from file '''
    with open(dataPath, 'r') as f:
        allData = [it[:-1] for it in f.readlines()]

    print(str(len(allData)) + ' frames loaded')
    for it, poseItem in enumerate(allData):
        # if it > 1000:
        #     break
        frameData = {}
        frameData['clipTitle'] = poseItem.split(',')[0]
        # use 256*256 resolution
        imgPose = np.array([float(it) for it in poseItem.split(',')[1:]]) # * 4 no need for hg_img
        imgPose.resize((16,2))
        frameData['imgPose'] = imgPose
        
        structuredData.append(frameData)
    
    clipTitles = set([x['clipTitle'][:-8] for x in structuredData])
    print(str(len(clipTitles)) + ' clip found')    
    return structuredData, clipTitles

''' Create images with 1 frame of pose estimation '''
def pose2Img(framePose, frameTitle):
    # im = Image.new('RGBA', (256, 256), (255, 255, 255, 0))
    imPath = '/data2/gengshan/tmp/' + frameTitle[:-8] + '/' + frameTitle + '.jpg'
    im = Image.open(imPath)
    draw = ImageDraw.Draw(im)
    for cord in framePose:
        draw.point(tuple(cord), 'red')
    for lineMark in pairRef:
        draw.line(tuple(framePose[lineMark[0]]) + \
                  tuple(framePose[lineMark[1]]), fill=1, width=3) 
    im.save("/home/gengshan/workJul/poseCaptioner_train/tmp/" + \
             frameTitle + ".jpg", "JPEG")
