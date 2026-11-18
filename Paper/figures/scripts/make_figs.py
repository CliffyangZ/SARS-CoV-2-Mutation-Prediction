from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qsim import *

FIG = Path(__file__).resolve().parent.parent
plt.rcParams.update({'font.size':8,'axes.linewidth':0.6,'figure.dpi':150,'pdf.fonttype':42,
  'axes.spines.top':False,'axes.spines.right':False})
C={'X':'#0072B2','Y':'#D55E00','Z':'#009E73'}

# ---- (b) readout vs z1 ----
n,T,tau=4,4,0.8
rng=np.random.default_rng(3)
zf=rng.uniform(-np.pi/2,np.pi/2,2*n)
grid=np.linspace(-np.pi/2,np.pi/2,241)
p=plus(n); main=[]
for z1 in grid:
    z=zf.copy(); z[0]=z1
    psi=Utrot(z,n,T,tau)@p
    zs=sum(bloch(psi,n,k)[2] for k in range(n)); assert abs(zs)<1e-9   # Prop. 1
    main.append(bloch(psi,n,0))
main=np.array(main)
c3=np.stack([np.cos(2*tau*grid),np.sin(2*tau*grid),0*grid],1)
# Prop. 2 check: no coupling -> closed form
z=zf.copy(); z[n:]=0; z[0]=0.7
assert np.abs(bloch(Utrot(z,n,T,tau)@p,n,0)-[np.cos(2*tau*.7),np.sin(2*tau*.7),0]).max()<1e-9
# RFF ridge surrogate of the 1-D slice
r2=np.random.default_rng(0); D=24
Om=r2.normal(0,2*tau,D); be=r2.uniform(0,2*np.pi,D)
F=lambda g:np.concatenate([np.cos(np.outer(g,Om)+be),np.sin(np.outer(g,Om)+be)],1)
tr=np.linspace(-np.pi/2,np.pi/2,60); idx=np.searchsorted(grid,tr).clip(0,len(grid)-1)
A=F(tr); W=np.linalg.solve(A.T@A+1e-3*np.eye(2*D),A.T@main[idx])
rff=F(grid)@W
rmse=np.sqrt(((rff-main)**2).mean())
fig,ax=plt.subplots(figsize=(2.4,2.1))
for i,k in enumerate('XYZ'):
    ax.plot(grid,main[:,i],color=C[k],lw=1.4,label=fr'$\langle {k}_1\rangle$ MAIN')
    ax.plot(grid,c3[:,i],color=C[k],lw=0.9,ls='--')
    ax.plot(grid,rff[:,i],color=C[k],lw=0.9,ls=':')
ax.set_xlabel(r'$z_1$'); ax.set_ylabel('readout of qubit 1')
ax.set_xticks([-np.pi/2,0,np.pi/2]); ax.set_xticklabels([r'$-\pi/2$','0',r'$\pi/2$'])
from matplotlib.lines import Line2D
h=[Line2D([],[],color='k',lw=1.4),Line2D([],[],color='k',lw=0.9,ls='--'),Line2D([],[],color='k',lw=0.9,ls=':')]
ax.legend([Line2D([],[],color=C[k]) for k in 'XYZ'],[r'$\langle X\rangle$',r'$\langle Y\rangle$',r'$\langle Z\rangle$'],fontsize=6,loc='upper right',ncol=3,frameon=False,columnspacing=0.8,handlelength=1)
ax.set_ylim(-1.15,1.45); ax.text(-0.28,1.02,'(b)',transform=ax.transAxes,fontweight='bold')
fig.tight_layout(pad=0.3); fig.savefig(FIG / 'fig_readout.pdf'); plt.close(fig)
print('RFF slice rmse',rmse, 'max|main-c3|',np.abs(main-c3).max())

# ---- (c) Trotter error ----
n,tau=3,0.7; Ts=[2,4,8,16,32,64,128,256]
rng=np.random.default_rng(0); zs=[rng.uniform(-np.pi/2,np.pi/2,2*n) for _ in range(100)]
Es=[expm(-1j*tau*H(z,n)) for z in zs]
mx=[];mn=[]
for T in Ts:
    e=[np.linalg.norm(Utrot(z,n,T,tau)-E,2) for z,E in zip(zs,Es)]
    mx.append(max(e)); mn.append(np.mean(e))
print(dict(zip(Ts,np.round(mx,5))))
fig,ax=plt.subplots(figsize=(2.4,2.1))
ax.loglog(Ts,mx,'o-',color='#0072B2',ms=3,lw=1.2,label='max over 100 draws')
ax.loglog(Ts,mn,'s-',color='#009E73',ms=3,lw=1.2,label='mean')
ax.loglog(Ts,mx[0]*Ts[0]/np.array(Ts),'k:',lw=0.9,label=r'$\propto 1/T$')
ax.axvline(4,color='#D55E00',lw=0.8,ls='--'); ax.text(4.3,ax.get_ylim()[0]*1.6,r'$T{=}4$ used',color='#D55E00',fontsize=6)
ax.set_xlabel(r'Trotter steps $T$'); ax.set_ylabel(r'$\|U_T-e^{-i\tau H}\|$')
ax.legend(fontsize=6,frameon=False); ax.text(-0.28,1.02,'(c)',transform=ax.transAxes,fontweight='bold')
fig.tight_layout(pad=0.3); fig.savefig(FIG / 'fig_trotter.pdf'); plt.close(fig)

# ---- (d) decision criteria ----
names=['MAIN','C1','C2','C3','C4','C5','C0']
f1=[0.7547,0.7534,0.7552,0.7544,0.7489,0.7481,0.7481]
fig,ax=plt.subplots(figsize=(2.4,2.1))
ax.axvspan(0.7547-0.02,0.7547+0.02,color='#CCCCCC',alpha=.5,lw=0)
ax.axvline(0.7547,color='k',lw=0.6)
y=np.arange(len(names))[::-1]
cols=['#0072B2']+['#555555']*6
ax.scatter(f1,y,c=cols,s=16,zorder=3)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlim(0.72,0.79); ax.set_xlabel('test macro-F1 (seed 0)')
ax.text(0.7547+0.001,len(names)-0.35,r'MAIN $\pm\delta$, $\delta{=}0.02$',fontsize=6,va='bottom')
ax.set_ylim(-0.6,len(names)+0.2); ax.text(-0.32,1.02,'(d)',transform=ax.transAxes,fontweight='bold')
fig.tight_layout(pad=0.3); fig.savefig(FIG / 'fig_decision.pdf'); plt.close(fig)
