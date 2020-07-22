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
curDir = './'
#curDir = '/scratch/fraphel/docksi_simexpAxion/code/'
chan = 'pot'

config = configparser.RawConfigParser()
config.read(curDir+'configDOCKSI.ini')

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
nbCond = 3

dataS = ioB.readLargeBin(dirDat+'dictio_Axion.bin')

dataSim = []
posNotNan = []
for line, idata in enumerate(dataS):
    if np.isnan(idata).any():
        print('There is a Nan')
    else:
        dataSim.append(idata)
        posNotNan.append(line)

dataSim = np.array(dataSim)
posNotNan = np.array(posNotNan)
nbSim = dataSim.shape[0]




drug = [[['A1', 'B1', 'C1', 'D1', 'E1'],['F1', 'G1', 'H1', 'A2', 'B2'],['C2', 'D2', 'E2', 'F2', 'G2'],['H2', 'A3', 'B3', 'C3', 'D3']] \
        ,[['E3', 'F3', 'G3', 'H3', 'A4'],['B4', 'C4', 'D4', 'E4', 'F4'],['G4', 'H4', 'A5', 'B5', 'C5'],['D5', 'E5', 'F5', 'G5', 'H5']] \
        ,[['F6', 'G6', 'H6', 'A7', 'B7'],['C7', 'D7', 'E7', 'F7', 'G7'],['H7', 'A8', 'B8', 'C8', 'D8'],['E8', 'F8', 'G8', 'H8', 'A9']] \
        ,[['B9', 'C9', 'D9', 'E9', 'F9'],['G9', 'H9', 'A10', 'B10', 'C10'],['D10', 'E10', 'F10', 'G10', 'H10'],['A11', 'B11', 'C11', 'D11', 'E11']] \
        ,[['A1', 'B1', 'C1', 'D1', 'E1'],['F1', 'G1', 'H1', 'A2', 'B2'],['C2', 'D2', 'E2'],['A3', 'B3', 'C3', 'D3']] \
        ,[['F3', 'G3', 'H3', 'A4'],['B4', 'C4', 'D4', 'E4', 'F4'],['G4', 'A5', 'B5', 'C5'],['D5', 'E5', 'F5', 'G5', 'H5']] \
        ,[['F6', 'G6', 'H6', 'A7', 'B7'],['C7', 'D7', 'E7', 'F7', 'G7'],['H7', 'A8', 'B8', 'C8', 'D8'],['E8', 'F8', 'G8', 'H8', 'A9']] \
        ,[['B9', 'C9', 'D9', 'E9', 'F9'],['G9', 'A10', 'B10', 'C10'],['D10', 'E10', 'F10', 'G10', 'H10'],['A11', 'B11', 'C11', 'D11', 'E11']] \
        ,[['A1', 'B1', 'C1', 'D1', 'E1'],['F1', 'G1', 'H1', 'A2', 'B2'],['C2', 'D2', 'E2', 'F2', 'G2'],['A3', 'B3', 'C3', 'D3', 'E3']] \
        ,[['F3', 'G3', 'H3', 'A4', 'B4'],['C4', 'D4', 'E4', 'F4', 'G4'],['H4', 'A5', 'B5', 'C5', 'D5'],['E5', 'F5', 'G5', 'H5', 'A6']] \
        ,[['B6', 'C6', 'D6', 'E6', 'F6'],['G6', 'H6', 'A7', 'B7', 'C7'],['D7', 'E7', 'F7', 'G7', 'H7'],['A8', 'B8', 'C8', 'D8', 'E9']] \
        ,[['F8', 'G8', 'H8', 'A9', 'B9'],['C9', 'D9', 'E9', 'F9', 'G9'],['H9', 'A10', 'B10', 'C10', 'D10'],['E10', 'F10', 'G10', 'H10', 'A11']]]



labelSim = []
for isim in posNotNan:
    thetaFile = np.loadtxt(fileLab+'theta_'+str(isim)+'.txt')
    associatedCond = thetaFile[1,1:1+nbCond]#thetaFile[irep*nbConc+2*iconc+1,1:1+nbCond]
    ilab = np.argmax(abs(1.0-associatedCond))
    labelSim.append(ilab)
labelSim = np.array(labelSim)
labLoc = []

labExp = [-1, 1, 1, 0, -1, 2, -1, 2, -1, -1, -1, -1]#-1: unknown
posUnknown = [0, 4, 6, 8, 9, 10, 11]
labExpKnown = []

if chan=='pot':
    labExpKnown = [1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1]# potassium
    for ilab in labelSim:
        if ilab==1:
            labLoc.append(1)
        else:
            labLoc.append(-1)
elif chan=='cal':
    labExpKnown = [1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1]# calcium
    for ilab in labelSim:
        if ilab==2:
            labLoc.append(1)
        else:
            labLoc.append(-1)
elif chan=='sod':
    labExpKnown = [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1]# sodium
    for ilab in labelSim:
        if ilab==0:
            labLoc.append(1)
        else:
            labLoc.append(-1)
labelSim = np.array(labLoc)


dataExpTrain = []; dataExpValid = []
labExpTrain = []; labExpValid = []

n_train = []
n_valid = []

for idrug in range(12):
    dirDrug = dirDat + 'drug' + str(idrug+1) + '/'

    if idrug in posUnknown:#Unknown
        cpt = 0
        for filename in os.listdir(dirDrug):
            well = filename[-7:-4]
            if well[0]=='_':
                well = well[1:]
            numConc = -1
            drugWell = drug[idrug]
            for iconc in range(len(drugWell)):
                if well in drugWell[iconc]:
                    numConc = iconc+1
                    dataExpValid.append(np.loadtxt(dirDrug+filename))
                    labExpValid.append(labExpKnown[idrug])
                    cpt += 1
        n_valid.append(cpt)    
    else:
        cpt = 0
        for filename in os.listdir(dirDrug):
            well = filename[-7:-4]
            if well[0]=='_':
                well = well[1:]
            numConc = -1
            drugWell = drug[idrug]
            for iconc in range(len(drugWell)):
                if well in drugWell[iconc]:
                    numConc = iconc+1
                    dataExpTrain.append(np.loadtxt(dirDrug+filename))
                    labExpTrain.append(labExpKnown[idrug])
                    cpt += 1
        n_train.append(cpt)


dataExpTrain = np.array(dataExpTrain)
dataExpValid = np.array(dataExpValid)
labExpTrain = np.array(labExpTrain)
labExpValid = np.array(labExpValid)

#print('SHAPES = ', dataExpTrain.shape, '', dataExpValid.shape, '', labExpTrain.shape, '', labExpValid.shape, '', n_train, '', n_valid)

lenTrainer = len(labelSim)+len(labExpTrain)

print('SHAPES = ', dataExpTrain.shape, '', dataExpValid.shape, '', labExpTrain.shape, '', labExpValid.shape, '', n_train, '', n_valid)
'''

data = np.vstack((dataSim, dataExpTrain))
data = np.vstack((data, dataExpValid))

labs = np.hstack((labelSim, labExpTrain))
labs = np.hstack((labs, labExpValid))

labels = np.array(labs)

for line, idata in enumerate(data):
    if np.isnan(idata).any():
        print('New.....There is a Nan')

print('shapes = ', data.shape, '', labels.shape, ' lenTrainer = ', lenTrainer)
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
        dirComp = curDir+'dim_'+str(dimProj)+'_comp'+str(icomp)+'/'
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
        dirNames = [f for f in os.listdir(curDir) if f.startswith(dirDim)]
        
        nbComp_dim = len(dirNames)
        posDim = np.zeros(nbComp_dim); betaDim = np.zeros(nbComp_dim)
        for icomp in range(1, nbComp_dim+1):
            dirNameComp = curDir+dirDim+'_comp'+str(icomp)+'/'
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
    mkdir_cmd = 'mkdir ' + curDir + dirOutPos
    os.system(mkdir_cmd)
    
    mv_cmd = 'mv ' + curDir + 'outcmaes_'+str(pos)+'_*.dat ./'+dirOutPos+'/'
    os.system(mv_cmd)

    dirComp = 'dim_'+str(dimProj)+'_comp'+str(numComp)

    if not os.path.isdir(dirComp):
        mkdir_cmd = 'mkdir ' + curDir + dirComp
        os.system(mkdir_cmd)

    mv_cmd = 'mv ' + curDir + 'pos'+str(pos)+' '+curDir+dirComp+'/'
    os.system(mv_cmd)
#--------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------
# Find the best component
def writeMinPos():
    dirComp = curDir+'dim_'+str(dimProj)+'_comp'+str(numComp)+'/'
    
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
    prefix = curDir+'outcmaes_'+str(pos)+'_'

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
'''

#--------------------------------------------------------------------------------------------------------
'''
vmin = sys.argv[1]
vmax = sys.argv[2]
vI = vecIndices[int(vmin):int(vmax)]
nbProc = int(sys.argv[3])

#vI = [60, 84, 89, 138]
#nbProc = int(sys.argv[1])

print('RUNNING ON ', nbProc, ' PROCS')
if __name__ == "__main__": 
  p = multiprocessing.Pool(nbProc)
  output = p.map(runMultiple,vI)
'''
writeMinPos()
#--------------------------------------------------------------------------------------------------------

