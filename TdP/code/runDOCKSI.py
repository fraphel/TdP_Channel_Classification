#from sklearn.svm import SVC
import numpy as np
import os, sys
import configparser

from computeScore_loc import scoreFunction
import cma
import ioBin as ioB
import multiprocessing

from os import listdir
from os.path import isfile, join
import re
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# Read configuration file

config = configparser.RawConfigParser()
config.read(sys.argv[1])

# CMAES
sigma = config.getfloat('CMAES', 'sigma')
tolfun = config.getfloat('CMAES', 'tolfun')

# DOCKSI
dimProj = config.getint('DOCKSI', 'dimProj')
numComp = config.getint('DOCKSI', 'numComp')
outputDim = config.getint('DOCKSI', 'outputDim')
nbProc = config.getint('DOCKSI', 'nbProcs')
lenTrainer = config.getint('DOCKSI', 'lenTrainer')
useCorrelation = config.getboolean('DOCKSI', 'useCorrelation')
threshold = config.getfloat('DOCKSI', 'threshold')

# Classification parameters
type = config.get('Classification', 'type')
kernel = config.get('Classification', 'kernel')
degree = config.getint('Classification', 'degree')
validation = config.get('Classification', 'validation')
dataType = config.get('Classification', 'dataType')

# Directories parameters
dirDat = config.get('Directories', 'dirDat')
fileLab = config.get('Directories', 'fileLab')
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# Read the data
labels = np.loadtxt(fileLab)
datafiles = [f for f in listdir(dirDat) if (isfile(join(dirDat, f)) and f.endswith('.bin'))]
print(datafiles)
allData_train = []; allLabels_train = []
allData_test = []; allLabels_test = []

for numfile, filename in enumerate(datafiles):
    print(numfile, '', dirDat+filename)
    data = ioB.readLargeBin(dirDat+filename)

    if numfile==0:
        allData_train = data[:76,:]
        allLabels_train = labels[:76]
        
        allData_test = data[76:,:]
        allLabels_test = labels[76:]
    else:
        allData_train = np.vstack((allData_train, data[:76,:]))
        allLabels_train = np.hstack((allLabels_train, labels[:76]))
        
        allData_test = np.vstack((allData_test, data[76:,:]))
        allLabels_test = np.hstack((allLabels_test, labels[76:]))
        
print(allData_train.shape, '', allLabels_train.shape)
print(allData_test.shape, '', allLabels_test.shape)

data = np.vstack((allData_train, allData_test))
labels = np.hstack((allLabels_train,allLabels_test))

print('shapes = ', data.shape, '', labels.shape)

#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# Remove undesired signal entries
'''
if dataType=='FP':
    data = np.delete(data, [38,39,40,41,42], axis=1)
elif dataType=='Ca':
    data = data[:,[38,39,40,41,42]]
'''
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------

classifier = {'type':type, 'kernel':kernel, 'degree':degree}


#--------------------------------------------------------------------------------------------------------
# Get the indices
vecIndices = range(data.shape[1])
bestR = []; beta_Init = []

if numComp==1:# CMAES DOES NOT WORK WITH SCALARS (=>GHOST VARIABLE)
    beta_Init = np.ones(2)
else:
    beta_Init = np.ones(numComp)

if numComp>1:
    # Get beta obtained for the previous component
    for icomp in range(1, numComp):
        dirComp = 'dim_'+str(dimProj)+'_comp'+str(icomp)+'/'
        datafiles = [f for f in listdir(dirComp) if isfile(join(dirComp, f))]
        bestR.append(int(re.findall('\d+', datafiles[0])[0]))
        if icomp==numComp-1:
            betaMin = np.loadtxt(dirComp+datafiles[0])
            if numComp==2:# CMAES DOES NOT WORK WITH SCALARS (=>GHOST VARIABLE)
                beta_Init[0] = betaMin[0]
            else:
               beta_Init[:-1] = betaMin
    bestR = np.array(bestR)


if useCorrelation:
    # Compute correlation matrix
    corrMat = np.corrcoef(np.transpose(data))
    rmVec = []

    listInd = []
    for ilv in vecIndices:
        lineCorr = corrMat[ilv,:]
        indV = np.argwhere(abs(lineCorr)>threshold)
        listInd.append(indV)

    commonMatches = []
    # Remove entries highly correlated
    for ie in len(listInd-1):
        if ie==0:
            commonMatches = set(listInd[ie]).intersection(listInd[ie+1])
        else:
            commonMatches = set(commonMatches).intersection(listInd[ie+1])
    bestR = np.hstack((bestR, commonMatches))

vecIndices = np.delete(vecIndices, bestR)


dimData = []
if dimProj>=2:
    for idim in range(1, dimProj):
        dirDim = 'dim_'+str(idim)
        dirNames = [f for f in os.listdir('./') if f.startswith(dirDim)]
        
        nbComp_dim = len(dirNames)
        posDim = np.zeros(nbComp_dim); betaDim = np.zeros(nbComp_dim)
        for icomp in range(1, nbComp_dim+1):
            dirNameComp = dirDim+'_comp'+str(icomp)+'/'
            datafiles = [f for f in listdir(dirNameComp) if isfile(join(dirNameComp, f))]
            posDim[icomp-1] = int(re.findall('\d+', datafiles[0])[0])
            
            if icomp==nbComp_dim:
                betaDim = np.loadtxt(dirNameComp+datafiles[0])
        dimInfo = np.vstack((posDim, betaDim))
        dimData.append(dimInfo)
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------
# Post-processing
def writeOutput(pos):
    
    dirOutPos = 'pos'+str(pos)
    mkdir_cmd = 'mkdir ' + dirOutPos
    os.system(mkdir_cmd)
    
    mv_cmd = 'mv outcmaes_'+str(pos)+'*.dat ./'+dirOutPos+'/'
    os.system(mv_cmd)

    dirComp = 'dim_'+str(dimProj)+'_comp'+str(numComp)

    if not os.path.isdir(dirComp):
        mkdir_cmd = 'mkdir ' + dirComp
        os.system(mkdir_cmd)

    mv_cmd = 'mv pos'+str(pos)+' ./'+dirComp+'/'
    os.system(mv_cmd)
#--------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------
# Find the best component
def writeMinPos():
    dirComp = 'dim_'+str(dimProj)+'_comp'+str(numComp)+'/'
    
    fitness = []; beta = []
    for ii in vecIndices:
        fileName = dirComp+'pos'+str(ii)+'/outcmaes_'+str(ii)+'_xrecentbest.dat'
        if isfile(fileName):
            dat = np.loadtxt(fileName, skiprows=1)
            if len(dat)==0:
                dat = np.loadtxt(fileName)
            
            if len(dat.shape)==2:
                fitness.append(dat[-1,4])
                beta.append(dat[-1,5:])
            else:
                fitness.append(dat[4])
                beta.append(dat[5:])
        else:
            fitness.append(1000.)
            lenBeta = len(beta[-1])
            beta.append([0.]*lenBeta)

    fitness = np.array(fitness)
    beta = np.array(beta)

    posMin = np.argmin(fitness)
    betaMin = beta[posMin,:]

    #outfile = 'minDat_'+str(posMin)+'.txt'
    outfile = 'minDat_'+str(vecIndices[posMin])+'.txt'
    
    np.savetxt(dirComp+outfile, betaMin)
#--------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------
# Run optimization process for one component on one proc
def runMultiple(pos):
    
    posTest = np.hstack((bestR, pos))
    prefix = 'outcmaes_'+str(pos)+'_'

    cmaOpt = {'tolfun': tolfun, 'verb_filenameprefix': prefix}
    
    # Optimization process
    if numComp>1:
        res = cma.fmin(scoreFunction,beta_Init,sigma, options=cmaOpt, args=(data, labels, dimProj, outputDim, classifier, posTest, dimData, lenTrainer))
        print(res)
        writeOutput(pos)
    else:
        sz = len(np.unique(data[:,pos]))
        if sz>1:
            res = cma.fmin(scoreFunction,beta_Init,sigma, options=cmaOpt, args=(data, labels, dimProj, outputDim, classifier, posTest, dimData, lenTrainer))
            print(res)
            writeOutput(pos)
#--------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------

if __name__ == "__main__": 
  p = multiprocessing.Pool(nbProc)
  output = p.map(runMultiple,vecIndices)
  writeMinPos()

#--------------------------------------------------------------------------------------------------------

