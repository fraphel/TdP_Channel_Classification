import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import fig_module as fm
import itertools

sns.set()
sns.set_context('talk')
sns.set_style('white')

figure_width = 7.
figure_height = 6.

pos = [ [43, 76, 98], [96, 95, 67] ]
col = ['#3b5998', '#8b9dc3', '#8b9dc3', '#8b9dc3']
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# Prepare data
data, labels = fm.getData('../../tdp.txt', '../../dictio/')
print('LABELS = ', labels.shape)
entries1, entries2, entries3 = fm.getEntries(data.shape)
wR1 = np.loadtxt('../saveRes/dim_1_comp3/minDat_98.txt')
wR2 = np.loadtxt('../saveRes/dim_2_comp3/minDat_67.txt')
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#

score_, valid_, y_pred_full_tr, y_pred_full_va = fm.getScoreValidation(pos, data, labels)
print('Prediction shape = ', y_pred_full_tr.shape, '', y_pred_full_va.shape)
y_true_full_tr = labels[:1520]
y_true_full_va = labels[1520:]

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# First figure: input space
beta_R1 = np.zeros(data.shape[1])
beta_R2 = np.zeros(data.shape[1])

beta_R1[pos[0]] = wR1
beta_R2[pos[1]] = wR2

comp1 = fm.getInput(beta_R1, data)
comp2 = fm.getInput(beta_R2, data)

inpVec = np.transpose(np.vstack((comp1, comp2)))

v_tdp = []; v_notdp = []
for ilab, iv in zip(labels, inpVec):
    if ilab==1:
        v_tdp.append(iv)
    else:
        v_notdp.append(iv)
v_tdp = np.array(v_tdp)
v_notdp = np.array(v_notdp)

colTdP = '#e0301e'
colNoTdP = '#005b96'

fig1 = plt.figure(figsize=[figure_width,figure_height])
ax_1 = fig1.add_subplot(111)
ax_1.scatter(v_tdp[:,0], v_tdp[:,1], c=colTdP, label='TdP risk')
ax_1.scatter(v_notdp[:,0], v_notdp[:,1], c=colNoTdP, label='NoTdP risk')

ax_1.set_xlabel('First dimension')
ax_1.set_ylabel('Second dimension')

ax_1.legend()
ax_1.grid()
fig1.tight_layout(pad=.3,w_pad=0.,h_pad=0.)
#fig1.savefig('/local/fraphel/publireo/IJNMBE_TdP_2017/version_06_2018/docksi_figs/input_comp.png',bbox_inches='tight')
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# Second figure: Weights
fig2 = plt.figure(figsize=[figure_width,figure_height])
ax1_2 = fig2.add_subplot(311)
ax2_2 = fig2.add_subplot(312)
ax3_2 = fig2.add_subplot(313)

w1_1 = np.zeros(len(entries1))
w1_2 = np.zeros(len(entries2))
w1_3 = np.zeros(len(entries3))

w2_1 = np.zeros(len(entries1))
w2_2 = np.zeros(len(entries2))
w2_3 = np.zeros(len(entries3))

for num, ip in enumerate(pos[0]):
    if ip<len(entries1):
        w1_1[ip] = wR1[num]
    elif ip<len(entries1)+len(entries2):
        w1_2[ip-len(entries1)] = wR1[num]
    else:
        w1_3[ip-(len(entries1)+len(entries2))] = wR1[num]

for num, ip in enumerate(pos[1]):
    if ip<len(entries1):
        w2_1[ip] = wR2[num]
    elif ip<len(entries1)+len(entries2):
        w2_2[ip-len(entries1)] = wR2[num]
    else:
        w2_3[ip-(len(entries1)+len(entries2))] = wR2[num]


ax1_2.plot(entries1, w1_1)
ax1_2.plot(entries1, w2_1)
plt.setp(ax1_2.xaxis.get_majorticklabels(), rotation=90)

ax2_2.plot(entries2, w1_2)
ax2_2.plot(entries2, w2_2)

ax3_2.plot(entries3, w1_3)
ax3_2.plot(entries3, w2_3)


ax1_2.grid()
ax2_2.grid()
ax3_2.grid()


fig2.tight_layout(pad=.3,w_pad=0.,h_pad=0.)
#fig2.savefig('/local/fraphel/publireo/IJNMBE_TdP_2017/version_06_2018/docksi_figs/weightcomp.png',bbox_inches='tight')
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# Third figure: score vs components
fig3 = plt.figure(figsize=[figure_width,figure_height])
ax_3 = fig3.add_subplot(111)

ax_3.plot(score_, valid_, c=col[0])
ax_3.scatter(score_, valid_, c=col[0])


ax_3.text(score_[0]+0.01, valid_[0]+1e-3, '$\mathbb{R}_1$')
ax_3.text(score_[1]+0.01, valid_[1]+1e-3, '$\mathbb{R}_2$')
ax_3.text(score_[2]+0.01, valid_[2]+1e-3, '$\mathbb{R}_3$')
ax_3.text(score_[3]+0.01, valid_[3]+1e-3, '$\mathbb{R}^2_1$')
ax_3.text(score_[4]+0.01, valid_[4]+1e-3, '$\mathbb{R}^2_2$')
ax_3.text(score_[5]-0.03, valid_[5]+2e-3, '$\mathbb{R}^2_3$')


ax_3.set_xlabel('Score')
ax_3.set_ylabel('Validation success')

ax_3.legend()
ax_3.grid()
fig3.tight_layout(pad=.3,w_pad=0.,h_pad=0.)
#fig3.savefig('/local/fraphel/publireo/IJNMBE_TdP_2017/version_06_2018/docksi_figs/scorevalid.png',bbox_inches='tight')
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# Fourth/Fifth figure: Confusion matrices
np.set_printoptions(precision=2)
#class_names = ['Non TdP risk', 'TdP Risk']
class_names = ['No', 'Yes']

tick_marks = np.arange(len(class_names))
solClass = ['', class_names[0], class_names[1], '']
fmt = '.2f'

cmap_tr=plt.cm.Blues
cmap_va=plt.cm.Reds

fig4, axes = plt.subplots(y_pred_full_tr.shape[0], 2, sharey='row', sharex=True, figsize=[5,11])
fig5 = plt.figure(figsize=[figure_width,figure_height])

fig6 = plt.figure(figsize=[int(0.5*figure_width), int(0.5*figure_height)])
fig7 = plt.figure(figsize=[int(0.5*figure_width), int(0.5*figure_height)])
ax6 = fig6.add_subplot(111)
ax7 = fig7.add_subplot(111)

ax5_1 = fig5.add_subplot(121)
ax5_2 = fig5.add_subplot(122)

for ie in range(y_pred_full_tr.shape[0]):
    cm_tr, sew_tr, accw_tr, spw_tr = fm.getConfMatInfo(y_true_full_tr, y_pred_full_tr[ie])
    cm_va, sew_va, accw_va, spw_va = fm.getConfMatInfo(y_true_full_va, y_pred_full_va[ie])
    
    im_tr = axes[ie,0].matshow(cm_tr, cmap=cmap_tr, vmin=0, vmax=1)
    im_va = axes[ie,1].matshow(cm_tr, cmap=cmap_va, vmin=0, vmax=1)

    if ie==0:
        axes[ie,0].set_title('Predicted label')
        axes[ie,1].set_title('Predicted label')

    if ie==y_pred_full_tr.shape[0]-1:
        axes[ie,0].set_xticklabels(solClass)
        axes[ie,1].set_xticklabels(solClass)
        axes[ie,0].xaxis.set_ticks_position('bottom')
        axes[ie,1].xaxis.set_ticks_position('bottom')
    
    axes[ie,0].set_yticklabels(solClass)#, rotation=45)
    axes[ie,1].set_yticklabels(solClass)#, rotation=45)

    thresh_tr = cm_tr.max() / 2.
    thresh_va = cm_va.max() / 2.

    for i, j in itertools.product(range(cm_tr.shape[0]), range(cm_tr.shape[1])):
        axes[ie,0].text(j, i, format(cm_tr[i, j], fmt),
                   horizontalalignment="center",
                   color="white" if cm_tr[i, j] > thresh_tr else "black")
    axes[ie,0].set_ylabel('True label')
    

    for i, j in itertools.product(range(cm_va.shape[0]), range(cm_va.shape[1])):
        axes[ie,1].text(j, i, format(cm_va[i, j], fmt),
                   horizontalalignment="center",
                   color="white" if cm_va[i, j] > thresh_va else "black")

    if ie==y_pred_full_tr.shape[0]-1:
        im_tr = ax5_1.matshow(cm_tr, cmap=cmap_tr, vmin=0, vmax=1)
        im_va = ax5_2.matshow(cm_tr, cmap=cmap_va, vmin=0, vmax=1)
        ax5_1.set_title('Predicted label')
        ax5_2.set_title('Predicted label')
        ax5_1.set_xticklabels(solClass)
        ax5_2.set_xticklabels(solClass)
        ax5_1.xaxis.set_ticks_position('bottom')
        ax5_2.xaxis.set_ticks_position('bottom')
        
        ax5_1.set_yticklabels(solClass)#, rotation=45)
        ax5_2.set_yticklabels(solClass)#, rotation=45)


        # split fig
        im_tr6 = ax6.matshow(cm_tr, cmap=cmap_tr, vmin=0, vmax=1)
        im_va7 = ax7.matshow(cm_tr, cmap=cmap_va, vmin=0, vmax=1)
        ax6.set_title('Predicted label')
        ax7.set_title('Predicted label')
        ax6.set_xticklabels(solClass)
        ax7.set_xticklabels(solClass)
        ax6.xaxis.set_ticks_position('bottom')
        ax7.xaxis.set_ticks_position('bottom')
        
        ax6.set_yticklabels(solClass)#, rotation=45)
        ax7.set_yticklabels(solClass)#, rotation=45)
        


        
        thresh_tr = cm_tr.max() / 2.
        thresh_va = cm_va.max() / 2.
        
        for i, j in itertools.product(range(cm_tr.shape[0]), range(cm_tr.shape[1])):
            ax5_1.text(j, i, format(cm_tr[i, j], fmt),
                            horizontalalignment="center",
                            color="white" if cm_tr[i, j] > thresh_tr else "black")
            ax6.text(j, i, format(cm_tr[i, j], fmt),
                            horizontalalignment="center",
                            color="white" if cm_tr[i, j] > thresh_tr else "black")#, fontsize=16)
        ax5_1.set_ylabel('True label')
        ax6.set_ylabel('True label')
        
        
        for i, j in itertools.product(range(cm_va.shape[0]), range(cm_va.shape[1])):
            ax5_2.text(j, i, format(cm_va[i, j], fmt),
                            horizontalalignment="center",
                            color="white" if cm_va[i, j] > thresh_va else "black")
            ax7.text(j, i, format(cm_va[i, j], fmt),
                            horizontalalignment="center",
                            color="white" if cm_va[i, j] > thresh_va else "black")#, fontsize=16)
        ax7.set_ylabel('True label')


fig4.subplots_adjust(hspace=0.22)
fig4.tight_layout(pad=.3,w_pad=0.5,h_pad=0.4)
fig5.tight_layout(pad=.3,w_pad=0.5,h_pad=0.)

fig6.tight_layout(pad=.3,w_pad=0.5,h_pad=0.)
fig7.tight_layout(pad=.3,w_pad=0.5,h_pad=0.)

#fig4.savefig('/home/local/fraphel/publireo/ncardia_ECR/figs/app_tdp/all_confmat.png',bbox_inches='tight')
#fig5.savefig('/home/local/fraphel/publireo/ncardia_ECR/figs/app_tdp/best_confmat.png',bbox_inches='tight')

#fig6.savefig('/home/local/fraphel/publireo/ncardia_ECR/figs/app_tdp/best_confmat_train.png',bbox_inches='tight')
#fig7.savefig('/home/local/fraphel/publireo/ncardia_ECR/figs/app_tdp/best_confmat_valid.png',bbox_inches='tight')





plt.show()
