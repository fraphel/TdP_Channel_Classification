import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier


def computeScore(pred_, prob_, lab, nbPerClass, classes):
    sc = 0.0
    tt = 0
    for it in range(len(lab)):
        vv = np.equal(classes, lab[it])
        pos = np.argwhere(vv==True)[0]
        #print('classes = ', classes, ' nbpc = ', nbPerClass, ' pos = ', pos, ' lab = ', lab[it], ' ', prob_[it])
        if pred_[it]==lab[it]:
            sc += len(lab)*prob_[it]/nbPerClass[pos]
            tt += 1
        else:
            sc -= len(lab)*prob_[it]/nbPerClass[pos]
    sc /= len(lab)
    print('succ rate = ', tt/len(lab))
    return -sc

#---------------------------------
# Bootstrap Sampling algorithm

def runBS(input, output, classifier, oDim, lenTrainer):

    
    inputClass = input[:lenTrainer,:]
    inputValid = input[lenTrainer:,:]
    
    outputClass = output[:lenTrainer]
    outputValid = output[lenTrainer:]
    
    decision = []; success=0
    
    probs = np.zeros((len(outputClass), oDim))
    nt = np.zeros(len(outputClass))
    #preds = [[]]*len(outputClass) 

    random_state = np.random.RandomState(0)
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=np.power(2,10), random_state=random_state)
    
    for train, test in rskf.split(inputClass, outputClass):
        
        random_state = np.random.RandomState(0)
        clf = None

        if classifier['type']=='svm':
            clf = svm.SVC(kernel=classifier['kernel'], degree=classifier['degree'], probability=True, random_state=random_state)
        elif classifier['type']=='lda':
            clf = LDA(solver="svd", store_covariance=True)
        elif classifier['type']=='knn':
            clf = KNeighborsClassifier(5)

        probas_ = clf.fit(inputClass[train], outputClass[train]).predict_proba(inputClass[test])
        #pred_ = clf.fit(inputClass[train], outputClass[train]).predict(inputClass[test])
        '''
        print(probas_)
        print(pred_)
        print(outputClass[train])
        '''
        for ip, it in enumerate(test):
            probs[it,:] += probas_[ip,:]
            #preds[it].append(pred_[ip])
            nt[it] += 1.0

    # Validation part
    clf_v = None
    if classifier['type']=='svm':
        clf_v = svm.SVC(kernel=classifier['kernel'], degree=classifier['degree'], probability=True, random_state=random_state)
    elif classifier['type']=='lda':
        clf_v = LDA(solver="svd", store_covariance=True)
    elif classifier['type']=='knn':
        clf_v = KNeighborsClassifier(5)
        
    probas_v = clf_v.fit(inputClass, outputClass).predict_proba(inputValid)
    #pred_v = clf_v.fit(inputClass, outputClass).predict(inputValid)

    
    prob_v = []; pred_v = []
    for ipv in probas_v:
        prob_v.append(max(ipv))
        pred_v.append(np.argmax(ipv))
    prob_v = np.array(prob_v)
    pred_v = np.array(pred_v)
    
    '''
    prob_t = []; pred_t = []
    for ie in range(len(nt)):
        med = probs[it,:]/nt[it]
        prob_t.append(max(med))
        (values, counts) = np.unique(preds[it], return_counts=True)
        #pred_t.append(values[np.argmax(counts)])
        pred_t.append(np.argmax(med))
    prob_t = np.array(prob_t)
    pred_t = np.array(pred_t)
    '''
    score_t = np.zeros((len(outputClass), oDim))
    for ip in range(len(outputClass)):
        score_t[ip,:] = probs[ip,:]/nt[ip]
    
    pred_t = []; prob_t = []
    for il in score_t:
        posLab = np.argmax(il)
        prob_t.append(max(il))
        if posLab==0:
            pred_t.append(0)
        elif posLab==1:
            pred_t.append(1)
        else:
            pred_t.append(2)
    pred_t = np.array(pred_t)
    prob_t = np.array(prob_t)
    
    return pred_t, prob_t, pred_v, prob_v
#---------------------------------



#---------------------------------
# Compute cost function

def scoreFunction(beta, data, labels, dimProj, outputDim, classifier, posTest, dimData, lenTrainer):

    dictio = []

    # Prepare data for Scikit
    if dimProj==1:
        dummy = np.zeros(data.shape[0])
        for ie, ipos in enumerate(posTest):
            dummy += data[:,int(ipos)]*beta[ie]
        for idd in dummy:
            dictio.append([idd])
        dictio = np.array(dictio)
    else:
        for iproj in range(1, dimProj):
            idimDat = np.transpose(dimData[iproj-1])
            dummy = np.zeros(data.shape[0])
            for icompDat in idimDat:
                dummy += data[:, int(icompDat[0])]*icompDat[1]

            dimDic = []
            for idd in dummy:
                dimDic.append([idd])
            dimDic = np.array(dimDic)
            if iproj==1:
                dictio = dimDic
            else:
                dictio = np.hstack((dictio, dimDic))
                
        
        dummy = np.zeros(data.shape[0])
        for ie, ipos in enumerate(posTest):
            dummy += data[:,int(ipos)]*beta[ie]
        dimDic = []
        for idd in dummy:
            dimDic.append([idd])
        dimDic = np.array(dimDic)
        dictio = np.hstack((dictio, dimDic))

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    # Run Bootstrap Sampling
    pred_t, prob_t, pred_v, prob_v = runBS(dictio, labels, classifier, outputDim, lenTrainer)
    #--------------------------------------------------------------------------------------------------------
   
    (classes_t, nbPerClass_t) = np.unique(labels[:lenTrainer], return_counts=True)
    (classes_v, nbPerClass_v) = np.unique(labels[lenTrainer:], return_counts=True)
 
    sc_t = computeScore(pred_t, prob_t, labels[:lenTrainer], nbPerClass_t, classes_t)
    sc_v = computeScore(pred_v, prob_v, labels[lenTrainer:], nbPerClass_v, classes_v)
    print('score train = ', sc_t, ' score validation = ', sc_v)
    return sc_t
