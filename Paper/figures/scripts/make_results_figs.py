from pathlib import Path
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'axes.linewidth':0.6,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
FIG = Path(__file__).resolve().parent.parent
ROOT = FIG.parents[1]
R = str(ROOT / 'Results/quspike_pqk/pqk_%s__mut__seed0.json')
cfg=['MAIN','C1','C2','C3','C4','C5','C0']
D={c:json.load(open(R%c)) for c in cfg}
col={'MAIN':'#0072B2','C1':'#E69F00','C2':'#009E73','C3':'#D55E00','C4':'#CC79A7','C5':'#7f7f7f'}
# (a) mixing weight search
fig,axs=plt.subplots(1,2,figsize=(6.6,2.3),gridspec_kw={'width_ratios':[1,1.25]})
ax=axs[0]
xs=[0,1,2,3,4,5]
for c in cfg[:-1]:
    ss=D[c]['scale_search']; ax.plot(xs,[x['val_f1_macro'] for x in ss],'o-',ms=3,lw=1.6 if c=='MAIN' else 0.9,color=col[c],label=c,zorder=3 if c=='MAIN' else 2)
ax.set_xticks(xs); ax.set_xticklabels(['0','1/4','1/2','1','2','4'])
ax.set_xlabel(r'mixing weight $s$'); ax.set_ylabel('validation macro-F1')
ax.axhline(D['MAIN']['scale_search'][0]['val_f1_macro'],color='k',lw=0.6,ls=':')
ax.text(5,D['MAIN']['scale_search'][0]['val_f1_macro']+0.0015,r'$s{=}0$ (reference)',ha='right',fontsize=6)
ax.legend(fontsize=6,frameon=False,ncol=2,loc='lower left'); ax.text(-0.22,1.02,'(a)',transform=ax.transAxes,fontweight='bold')
# (b) group-wise test F1
ax=axs[1]
grp=[('delta','Delta (D)'),('b1','B.1 family (B)'),('others','Others (O)')]
w=0.8/len(cfg); order=['C0','MAIN','C1','C2','C3','C4','C5']
col['C0']='#000000'
for i,c in enumerate(order):
    v=[D[c]['group_metrics_test'][g] for g,_ in grp]
    ax.bar(np.arange(3)+(i-3)*w,v,w,color=col[c],label=c,linewidth=0)
ax.set_xticks(range(3)); ax.set_xticklabels([n for _,n in grp]); ax.set_ylabel('test macro-F1 within group')
ax.set_ylim(0.5,1.0); ax.legend(fontsize=6,frameon=False,ncol=7,loc='upper center',columnspacing=0.8,handlelength=0.9,bbox_to_anchor=(0.5,1.12))
ax.text(-0.18,1.02,'(b)',transform=ax.transAxes,fontweight='bold')
fig.tight_layout(pad=0.4); fig.savefig(FIG / 'fig_results.pdf')
for c in order: print(c,[round(D[c]['group_metrics_test'][g],3) for g,_ in grp])
