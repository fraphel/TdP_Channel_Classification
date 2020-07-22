import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
#from colored import fg, bg, attr

#colors =  ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'dark_green', 'dodger_blue_1', 'purple_4a', 'steel_blue_3']

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
    
    random_state = np.random.RandomState(0)
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=500, random_state=random_state)
    
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
        
        for ip, it in enumerate(test):
            probs[it,:] += probas_[ip,:]
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
    pred_v = clf_v.fit(inputClass, outputClass).predict(inputValid)
    

    # Compute success rate for the validation data
    sc_v = 0.0
    for ip, ip_v in enumerate(probas_v):
        if np.argmax(ip_v)==0 and outputValid[ip]==-1:
            sc_v += 1.0
        if np.argmax(ip_v)==1 and outputValid[ip]==1:
            sc_v += 1.0
    sc_v /= float(len(outputValid))
    print('len valid = ', len(outputValid), ' pred_v shape = ', pred_v.shape)
    print('VALIDATION: ')
    for ie in range(20):
        print('PRED = ', np.argmax(probas_v[10*ie:10*(ie+1)], axis=1))
    print('REAL = ', outputValid[:10].astype(int))
    
    score = np.zeros((len(outputClass), oDim))
    for ip in range(len(outputClass)):
        score[ip,:] = probs[ip,:]/nt[ip]
    
    predLab = []
    for il in score:
        posLab = np.argmax(il)
        if posLab==0:
            predLab.append(-1)
        else:
            predLab.append(1)
            
    predLab = np.array(predLab)
    
    for ilab in range(len(outputClass)):
        if predLab[ilab] == outputClass[ilab]:
            success += 1

        iscore = score[ilab]
        decision.append(np.max(iscore))
    
    percent = 100.*success/float(len(outputClass))
    
    return percent, np.array(decision), np.array(predLab), sc_v, pred_v
#---------------------------------



#---------------------------------
# Compute cost function

def scoreFunction(dictio, labels, classifier, beta, posTest, lenTrainer):
    print('posTest = ', posTest, ' beta = ', beta)
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    # Run Bootstrap Sampling
    percent, decision, labPred, sc_v, pred_v = runBS(dictio, labels, classifier, 2, lenTrainer)
    
    #--------------------------------------------------------------------------------------------------------
    
    # Compute new score:
    nbLab = len(labels)
    n_notdp = 0; n_tdp = 0
    for ilab in labels:
        if ilab>0:
            n_tdp += 1
        else:
            n_notdp += 1
    
    
    sc0 = 0.
    for id in range(len(decision)):
        if labels[id]<0:
            if labPred[id]<0:
                sc0 += decision[id]*(nbLab/n_notdp)
            else:
                sc0 -= decision[id]*(nbLab/n_notdp)
        else:
            if labPred[id]>0:
                sc0 += decision[id]*(nbLab/n_tdp)
            else:
                sc0 -= 2.*decision[id]*(nbLab/n_tdp) 
                
    print('len dec = ', len(decision), ' sc_v = ', sc_v)
    sc0 = sc0/float(len(decision))
    '''
    sumSquare = 0.0
    if len(posTest)==1:
        sumSquare = beta[0]*beta[0]
    else:
        for ib in beta:
            sumSquare += ib*ib
    diffOne = np.linalg.norm(1.0-sumSquare)
    
    sc0 -= 0.1*diffOne
    '''
    print('score = ', -sc0, ' validation = ', sc_v)
    return -sc0, sc_v, labPred, pred_v
