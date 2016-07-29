import h5py
import time

''' Count the number of clips and frames of this hdf5 file '''
def checkKeys(h5Path):
    h5File = h5py.File(h5Path, 'r')
    print('with ' + str(len(h5File.keys())) + ' clips')
    
    accum = 0
    for it in h5File.keys():
        accum += len(h5File[it].keys())
    print('with ' + str(accum) + ' frames')
    h5File.close()

''' merge the original h5 file to the target h5 file, using append mode '''
def mergeH5(originalPath, targetPath):
    ''' print info about target path '''
    print(originalPath + ' info:')
    checkKeys(originalPath)
    
    ''' open files and store keys '''
    newFile = h5py.File(targetPath)  # use 'a' may cause error, use default append instead
    oriFile = h5py.File(originalPath, 'r')  
    newClipTitles = newFile.keys()  # store it to avoid retriving every time
    oriClipTitles = oriFile.keys()  
    
    beg = time.time()
    ''' copy data '''
    for it, clipTitle in enumerate(oriClipTitles):
        if it % 100 == 0:
            print(str(it) + " finished")
            print('lapsed ' + str(time.time() - beg))
        if clipTitle in newClipTitles:  
            ''' add new datasets when current group exists '''
            toAdd = set(oriFile[clipTitle].keys()) - set(newFile[clipTitle].keys())
            for frameName in toAdd:
                datasetName = clipTitle + "/" + frameName
                h5py.h5o.copy(oriFile.id, datasetName, newFile.id, datasetName)
                
        else:
            ''' copy the whole group when with no current clip '''
            h5py.h5o.copy(oriFile.id, clipTitle, newFile.id, clipTitle)
            
    newFile.close()
    oriFile.close()

''' merge the original h5 file to the target h5 file, using external link '''
def mergeExtH5(originalPath, targetPath):
    ''' print info about target path '''
    print(originalPath + ' info:')
    checkKeys(originalPath)
    
    ''' open files and store keys '''
    newFile = h5py.File(targetPath)  # use 'a' may cause error, use default append instead
    oriFile = h5py.File(originalPath, 'r')  
    newClipTitles = newFile.keys()  # store it to avoid retriving every time
    oriClipTitles = oriFile.keys()
    
    beg = time.time()
    ''' copy data '''
    for it, clipTitle in enumerate(oriClipTitles):
        if it % 100 == 0:
            print(str(it) + " finished")
            print('lapsed ' + str(time.time() - beg))
        if clipTitle in newClipTitles:  
            ''' add new datasets when current group exists '''
            toAdd = set(oriFile[clipTitle].keys()) - set(newFile[clipTitle].keys())
            for frameName in toAdd:
                datasetName = clipTitle + "/" + frameName
                newFile[datasetName] = h5py.ExternalLink(originalPath, datasetName)
                
        else:
            ''' copy the whole group when with no current clip '''
            newFile[clipTitle] = h5py.ExternalLink(originalPath,clipTitle)
            
    newFile.close()
    oriFile.close()
