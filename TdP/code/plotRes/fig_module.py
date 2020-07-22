from os import listdir
from os.path import isfile, join
import ioBin as ioB
import numpy as np
from computeScore_loc import *
from sklearn.metrics import confusion_matrix

#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
# Get the data
def getData(fileLab, fileDic):

    labels = np.loadtxt(fileLab)
    
    datafiles = [f for f in listdir(fileDic) if (isfile(join(fileDic, f)) and f.endswith('.bin'))]
    allData_train = []; allLabels_train = []
    allData_test = []; allLabels_test = []

    for numfile, filename in enumerate(datafiles):
        data = ioB.readLargeBin(fileDic+filename)

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

    return data, labels

# Get entries
def getEntries(datShape):
    entries1 = ['med DA', 'med RA', 'med FPD', 'med AUC', 'med RM', 'med RW', 'med FPN',
            'mean DA', 'mean RA', 'mean FPD', 'mean AUC', 'mean RM', 'mean RW', 'mean FPN',
            'med DA/RA', 'med RA/DA', 'med RA/FPD', 'med FPD/RA', 'med DA/FPD', 'med FPD/DA', 'med RA/RW', 'med RW/RA',
            'mean DA/RA', 'mean RA/DA', 'mean RA/FPD', 'mean FPD/RA', 'mean DA/FPD', 'mean FPD/DA', 'mean RA/RW', 'mean RW/RA',
            'max DA/RA', 'max RA/DA', 'max RA/FPD', 'max FPD/RA', 'max DA/FPD', 'max FPD/DA', 'max RA/RW', 'max RW/RA',
            'CD 90', 'CD75', 'CD50', 'CD25', 'CA', 'RC', 'AUC90', 'AUC75', 'AUC50', 'AUC25',
            'CA*medFPD', 'CA*meanFPD', 'CA*CD90', 'CA*CD75', 'CA*CD50', 'CA*CD25',
            'medFPD*CD90', 'medFPD*CD75', 'medFPD*CD50', 'medFPD*CD25',
            'meanFPD*CD90', 'meanFPD*CD75', 'meanFPD*CD50', 'meanFPD*CD25']

    entries2 = ['K1', 'K2', 'K3', 'K4', 'K5']
    nb_wlt = datShape[1]-(len(entries1)+len(entries2))
    
    entries3 = range(nb_wlt)

    return entries1, entries2, entries3
#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
def getInput(beta, data):
    vChan = []
    for idata in data:
        val = 0.0
        for iid, iiw in zip(idata, beta):
            val += iid*iiw
        vChan.append(val)
    return np.array(vChan)
#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
def getScoreValidation(posChan, data, labels):
    score = []; valid = []
    y_pred_full_tr = []; y_pred_full_va = []
    
    for num, ipChan in enumerate(posChan):
        for icomp in range(len(ipChan)):
            comp = ipChan[icomp]
            allComp = ipChan[:icomp+1]
            
            input = []
            bloc = []
            
            if num==0:#R
                filename = '../saveRes/dim_'+str(num+1)+'_comp'+str(icomp+1)+'/minDat_'+str(comp)+'.txt'
                beta_loc = np.loadtxt(filename)
                
                if icomp==0:
                    beta_loc = np.array(beta_loc[-1])
                
                beta = np.zeros(data.shape[1])
                beta[allComp] = beta_loc
                
                input = getInput(beta, data)
                dummy = []
                for ii in input:
                    dummy.append([ii])
                input = np.array(dummy)
                bloc = [beta_loc]

            else:
                filename1 = '../saveRes/dim_1_comp'+str(len(posChan[0]))+'/minDat_'+str(posChan[0][-1])+'.txt'
                beta1_loc = np.loadtxt(filename1)

                beta1 = np.zeros(data.shape[1])
                beta1[posChan[0]] = beta1_loc

                filename = '../saveRes/dim_'+str(num+1)+'_comp'+str(icomp+1)+'/minDat_'+str(comp)+'.txt'
                beta2_loc = np.loadtxt(filename)
                
                if icomp==0:
                    beta2_loc = np.array(beta2_loc[-1])

                beta2 = np.zeros(data.shape[1])
                beta2[allComp] = beta2_loc
                
                input1 = getInput(beta1, data)
                input2 = getInput(beta2, data)

                input = np.transpose(np.vstack((input1, input2)))
                bloc = [beta2_loc]
            print('input shape = ', input.shape)
            classifier = {'type':'lda', 'kernel':'rbf', 'degree':1}
            isc, iva, ipred_tr, ipred_va = scoreFunction(input, labels, classifier, bloc, np.array(allComp), 1520)
            
            score.append(isc)
            valid.append(iva)
            y_pred_full_tr.append(ipred_tr)
            y_pred_full_va.append(ipred_va)
            
    score = np.array(score)
    valid = np.array(valid)
    y_pred_full_tr = np.array(y_pred_full_tr)
    y_pred_full_va = np.array(y_pred_full_va)
    
    return score, valid, y_pred_full_tr, y_pred_full_va
#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
# For confusion matrices
def tp(y_true, y_pred): return float(confusion_matrix(y_true, y_pred)[1, 1])
def tn(y_true, y_pred): return float(confusion_matrix(y_true, y_pred)[0, 0])
def fp(y_true, y_pred): return float(confusion_matrix(y_true, y_pred)[0, 1])
def fn(y_true, y_pred): return float(confusion_matrix(y_true, y_pred)[1, 0])


def getConfMatInfo(labTrue, labPred):
    cnf_matrix = confusion_matrix(labTrue, labPred)
    tp_val = tp(labTrue, labPred); tn_val = tn(labTrue, labPred)
    fp_val = fp(labTrue, labPred); fn_val = fn(labTrue, labPred)
    sew = round(tp_val/(tp_val+fn_val),2); spw = round(tn_val/(tn_val+fp_val),2)
    accw = round((tp_val+tn_val)/(tp_val+tn_val+fp_val+fn_val),2)
    cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    return cm, sew, accw, spw



#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#


