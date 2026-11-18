from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent
plt.rcParams.update({'font.size':8,'axes.linewidth':0.6,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
# (label, macro-F1, train minutes, family)   — values from the paper's results table / cnn_overview.md
E=[('One-hot RF',0.6974,35/60,'c'),('One-hot MLP',0.6978,19.78,'c'),('One-hot LR',0.7339,6.17,'c'),
   ('CNN3 + RP',0.6328,4988/60,'i'),('ResNet34 + RP',0.6633,9152/60,'i'),('CNN4 + RP',0.6787,5462/60,'i'),
   ('Linear ref. (C0)',0.7481,1/60,'q0'),('QuSpike-PQK',0.7547,3.09,'q')]
col={'c':'#999999','i':'#E69F00','q0':'#56B4E9','q':'#0072B2'}
fig,ax=plt.subplots(1,2,figsize=(6.6,2.3),sharey=True)
y=np.arange(len(E))
for a,idx,xl in ((ax[0],1,'test macro-F1'),(ax[1],2,'training time (min, log)')):
    a.barh(y,[e[idx] for e in E],color=[col[e[3]] for e in E],height=0.65)
    a.set_xlabel(xl)
ax[0].set_yticks(y); ax[0].set_yticklabels([e[0] for e in E]); ax[0].set_xlim(0.6,0.77)
for yy,e in zip(y,E): ax[0].text(e[1]+0.002,yy,f'{e[1]:.4f}',va='center',fontsize=6,fontweight='bold' if e[3]=='q' else None)
ax[1].set_xscale('log'); ax[1].set_xlim(0.01,500)
for yy,e in zip(y,E): ax[1].text(e[2]*1.25,yy,(f'{e[2]:.2f}' if e[2]>=1 else ('<1 s' if e[2]<0.05 else f'{e[2]*60:.0f} s')),va='center',fontsize=6,fontweight='bold' if e[3]=='q' else None)
ax[0].text(-0.62,1.02,'(a)',transform=ax[0].transAxes,fontweight='bold'); ax[1].text(-0.05,1.02,'(b)',transform=ax[1].transAxes,fontweight='bold')
fig.tight_layout(pad=0.4); fig.savefig(FIG / 'fig_compare.pdf')
