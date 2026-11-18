import numpy as np
from scipy.linalg import expm
I2=np.eye(2);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Zm=np.diag([1.,-1]).astype(complex)
def op(n,d):
    m=np.array([[1]],complex)
    for q in range(n): m=np.kron(m,d.get(q,I2))
    return m
def edges(n): return [(0,1)] if n==2 else [(j,(j+1)%n) for j in range(n)]
def H(z,n):
    h=sum(z[j]*op(n,{j:Zm}) for j in range(n))
    for k,(a,b) in enumerate(edges(n)):
        h=h+z[n+k]*sum(op(n,{a:P,b:P}) for P in (X,Y,Zm))
    return h
def Utrot(z,n,T,tau):
    D=tau/T; dim=2**n; step=np.eye(dim,dtype=complex)
    fields=np.eye(dim,dtype=complex)
    for j in range(n): fields=expm(-1j*D*z[j]*op(n,{j:Zm}))@fields
    step=fields
    for k,(a,b) in enumerate(edges(n)):
        h=sum(op(n,{a:P,b:P}) for P in (X,Y,Zm))
        step=expm(-1j*D*z[n+k]*h)@step
    return np.linalg.matrix_power(step,T)
def plus(n): 
    v=np.ones(2**n,complex)/np.sqrt(2**n); return v
def bloch(psi,n,k):
    return np.array([np.real(psi.conj()@op(n,{k:P})@psi) for P in (X,Y,Zm)])
