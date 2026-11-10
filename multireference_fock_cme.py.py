#!/usr/bin/env python3
"""Reproducible benchmark: Fock / polynomial occupations vs classical reductions.
All reduced methods advance from p(t=0) using known reaction generator; truth never used
in their evolution. Dynamics are finite-state CME, no statistical fitting.
"""
import os, sys, time, itertools, json, argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.special import comb, gammaln, logsumexp
from scipy.sparse import csr_matrix, diags, kron, eye
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import qr, svd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

OUT=os.path.join(os.path.dirname(__file__),'resultados')
os.makedirs(OUT,exist_ok=True)

def hs(x,mid,power=4):
    return x**power/(mid**power + x**power)

def build_network(d,L,scenario):
    nstates=L**d
    coords=np.indices((L,)*d,dtype=np.int16).reshape(d,-1)
    b0=np.full(d,.35); auto=np.full(d,1.3); neighbor=np.full(d,1.2); death=np.full(d,.55)
    positive=np.ones(d,dtype=bool); power=3; t_end=1.2
    if scenario=='independent':
        b0[:]=.9;auto[:]=0.;neighbor[:]=0.;death[:]=.55;power=2;t_end=1.3
    elif scenario=='weak':
        b0[:]=.7;auto[:]=.22;neighbor[:]=.25;death[:]=.5; power=2; t_end=1.5
    elif scenario=='bistable':
        b0[:]=.13;auto[:]=5.8;neighbor[:]=1.3;death[:]=.54;power=4; t_end=1.4
    elif scenario=='cascade':
        b0[:]=.16;auto[:]=.5;neighbor[:]=2.6;death[:]=.58;power=3; t_end=1.8
        positive[1::2]=False
    elif scenario=='stiff':
        b0[:]=.32;auto[:]=2.7;neighbor[:]=1.8;death[:]=.72;power=3; t_end=.95
        positive[1::2]=False
    else: raise ValueError(scenario)
    mid=(L-1)/2
    maxn=L-1
    levels=np.arange(L,dtype=float)
    H=hs(levels,mid,power)
    factors=(maxn-levels)/maxn
    terms=[]
    # rate b_i = cap(n_i) [b0 + auto*Hill(n_i) + neighbor*Hill(n_prev)]
    # or neighbor*(1-Hill) for inhibitory edges
    I=eye(L,format='csr')
    for i in range(d):
        prev=(i-1)%d
        fi=factors
        for kind,weight,other in [('constitutive',b0[i]*fi,None),
                                  ('self',auto[i]*fi*H,None),
                                  ('neighbor',neighbor[i]*fi,prev)]:
            if kind=='neighbor':
                g=H if positive[i] else 1-H
            loc=diags([-weight,weight[:-1]],[0,-1],shape=(L,L),format='csr')
            locs=[I]*d
            locs[i]=loc
            if other is not None: locs[other]=diags(g,0,format='csr')
            terms.append(locs)
        de=death[i]*levels
        locd=diags([-de,de[1:]],[0,1],shape=(L,L),format='csr')
        locs=[I]*d;locs[i]=locd
        terms.append(locs)
    # Full sparse generator using transition rates derived from identical reactions.
    idx=np.arange(nstates,dtype=np.int64)
    strides=[L**(d-1-i) for i in range(d)]
    rows=[];cols=[];data=[]
    out=np.zeros(nstates)
    for i in range(d):
        x=coords[i].astype(float)
        hp=hs(coords[(i-1)%d].astype(float),mid,power)
        neigh=hp if positive[i] else 1-hp
        hself=hs(x,mid,power)
        birth=((maxn-x)/maxn)*(b0[i]+auto[i]*hself+neighbor[i]*neigh)
        dec=death[i]*x
        ib=birth>0;idd=dec>0
        rows.extend([idx[ib]+strides[i],idx[idd]-strides[i]])
        cols.extend([idx[ib],idx[idd]])
        data.extend([birth[ib],dec[idd]])
        out+=birth+dec
    rows.append(idx);cols.append(idx);data.append(-out)
    Q=csr_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(nstates,nstates))
    Q.sum_duplicates()
    params=dict(d=d,L=L,scenario=scenario,b0=b0.tolist(),auto=auto.tolist(),neighbor=neighbor.tolist(),death=death.tolist(),positive=positive.tolist(),Hill_power=power,cap_birth=True)
    return Q,terms,coords,params,t_end

def initial_p(d,L,scenario,coords):
    # Pair of reproducible correlated product-binomial macrostates.
    n=coords.shape[1]
    comps=[]
    if scenario in ('weak','independent'):
        thetas=[np.full(d,.34)]; weights=[1.]
    elif scenario=='cascade':
        th=np.full(d,.13); th[0]=.88; th[1]=.55
        thetas=[th];weights=[1.]
    elif scenario=='bistable':
        thetas=[np.full(d,.12),np.full(d,.82)];weights=[.55,.45]
    elif scenario=='stiff':
        a=np.array([.1 if i%2==0 else .83 for i in range(d)])
        b=1.-a
        thetas=[a,b];weights=[.6,.4]
    for theta in thetas:
        p=np.ones(n)
        for i,t in enumerate(theta):
            xx=coords[i]
            p*=comb(L-1,xx)*t**xx*(1-t)**(L-1-xx)
        comps.append(p)
    result=np.array(weights)@np.stack(comps)
    return result/result.sum(),np.array(thetas),np.array(weights)

def orthogonal_local(L,mean=0.47):
    x=np.arange(L,dtype=float)
    weights=comb(L-1,x)*mean**x*(1-mean)**(L-1-x)
    z=(x-(L-1)*mean)/(np.sqrt((L-1)*mean*(1-mean)))
    V=np.column_stack([z**k for k in range(L)])
    B0=np.sqrt(weights[:,None])*V
    Q,R=np.linalg.qr(B0)
    for i in range(L):
        if np.dot(Q[:,i],B0[:,i])<0: Q[:,i]*=-1
    return Q

def total_degree_indices(d,L,K,max_support=None):
    result=[]
    for tup in itertools.product(range(min(K,L-1)+1),repeat=d):
        if sum(tup)<=K and (max_support is None or sum(a>0 for a in tup)<=max_support):
            result.append(tup)
    return sorted(result,key=lambda v:(sum(v),sum(x>0 for x in v),v))

def get_basis(d,L,K,mean=.47,max_support=None,local_basis=None):
    inds=total_degree_indices(d,L,K,max_support)
    local=orthogonal_local(L,mean) if local_basis is None else local_basis
    shape=(L,)*d
    n=L**d
    B=np.empty((n,len(inds)),dtype=np.float64,order='F')
    for j,alpha in enumerate(inds):
        a=np.array([1.])
        for x in alpha: a=np.multiply.outer(a,local[:,x]).ravel()
        B[:,j]=a
    return B,inds

def check_Q(Q):
    e=np.asarray(Q.sum(axis=0)).ravel()
    assert max(abs(e))<5e-12,max(abs(e))
    assert Q.diagonal().max()<=1e-12


def expm_truth(Q,p0,times):
    t0=time.perf_counter()
    mat=expm_multiply(Q,p0,start=0.,stop=times[-1],num=len(times),endpoint=True)
    return np.asarray(mat),time.perf_counter()-t0

def predict_meanfield(Q,coords,params,thetas,weights,times):
    # independent binomial marginals propagated using a standard mean-field closure
    # no full vector Q or full-space fitting in the time evolution.
    t0=time.perf_counter()
    d=params['d'];L=params['L'];h=params['Hill_power'];M=L-1
    n=np.arange(L,dtype=float)
    hh=hs(n,M/2,h)
    cap=(M-n)/M
    init=[]
    for x in thetas:
        init.extend([comb(M,n)*x[i]**n*(1-x[i])**(M-n) for i in range(d)])
    def func(t,z):
        z=z.reshape(len(thetas),d,L)
        res=np.zeros_like(z)
        for j in range(len(thetas)):
            for i in range(d):
                ni=(i-1)%d
                hbar=hh@z[j,ni]
                hbar=hbar if params['positive'][i] else 1-hbar
                b=cap*(params['b0'][i]+params['auto'][i]*hh+params['neighbor'][i]*hbar)
                death=params['death'][i]*n
                p=z[j,i]
                res[j,i]-=(b+death)*p
                res[j,i,1:]+=b[:-1]*p[:-1]
                res[j,i,:-1]+=death[1:]*p[1:]
        return res.ravel()
    sol=solve_ivp(func,(0.,times[-1]),np.array(init).ravel(),t_eval=times,rtol=2e-7,atol=2e-9)
    if not sol.success: raise RuntimeError(sol.message)
    preds=[]
    for z in sol.y.T:
        z=z.reshape(len(thetas),d,L)
        p=np.zeros(L**d)
        for j in range(len(thetas)):
            pp=np.array([1.])
            for i in range(d):pp=np.multiply.outer(pp,z[j,i]).ravel()
            p+=weights[j]*pp
        preds.append(np.maximum(p,0)/sum(np.maximum(p,0)))
    return np.array(preds),time.perf_counter()-t0

def predict_linear(Q,p0,B,times):
    # Petrov/Galerkin on linear p = B c, with polynomial orthogonal basis.
    # no truth used. B columns Euclidean orthonormal.
    t0=time.perf_counter(); c0=B.T@p0
    QB=Q@B
    A=B.T@QB
    coeff=expm_multiply(A,c0,start=0.,stop=times[-1],num=len(times),endpoint=True)
    raw=coeff@B.T
    preds=np.maximum(raw,0)
    preds/=np.maximum(preds.sum(axis=1,keepdims=True),1e-300)
    elapsed=time.perf_counter()-t0
    return preds,raw,elapsed

def predict_amp(Q,p0,B,times,eps_factor=1.e-5,max_step=.025):
    # Direct projected sqrt-probability CME. Probability is ALWAYS nonnegative.
    # sqrt(p) = B c; p = (Bc)^2 / ||c||^2; exact sqrt dynamics:
    # d sqrt(p)/dt = (Qp)/(2 sqrt(p)) (if positive). We use regularized
    # tangent projection c'= B^T (Qp*psi)/(2*(psi^2+eps)); normalize c.
    t0=time.perf_counter(); n=len(p0)
    eps=eps_factor/n
    c0=B.T@np.sqrt(p0)
    c0=c0/max(np.linalg.norm(c0),1e-15)
    def f(c):
        psi=B@c
        pp=psi*psi
        pp/=np.dot(c,c)
        r=Q@pp
        # regularized 1/psi with symmetric limiting behavior
        v=.5*r*psi/(psi*psi+eps)
        dc=B.T@v
        dc-=c*np.dot(c,dc)/np.dot(c,c)
        return dc
    c=c0.copy(); preds=[]
    for tprev,t in zip([0.]+list(times[:-1]),times):
        if t>tprev:
            dt=t-tprev
            steps=int(np.ceil(dt/max_step))
            h=dt/steps
            for _ in range(steps):
                k1=f(c)
                trial=c+h*k1
                k2=f(trial)
                c+=.5*h*(k1+k2)
                c/=max(np.linalg.norm(c),1e-15)
        psi=B@c
        p=psi**2;p/=p.sum()
        preds.append(p)
    return np.array(preds),time.perf_counter()-t0

def predict_entropy(Q,p0,B,times,regularize=1.e-6,max_step=.025):
    # Normalized exponential family relative to product binomial reference rho:
    # p_theta(x)=rho(x) exp(sum theta_j G_j(x))/Z(theta), G_j=B_j/sqrt(rho).
    # Covariance Fisher matrix C; exact moment projection m'_j=E_p[L G_j].
    t0=time.perf_counter();N,M=B.shape
    rho=B[:,0]**2; rho/=rho.sum();rho=np.maximum(rho,1e-300)
    g=B[:,1:]/np.sqrt(rho)[:,None]
    scales=np.maximum(np.max(np.abs(g),axis=0),1e-15);g=g/scales
    lg=Q.T@g; k=g.shape[1];logrho=np.log(rho)
    moment=p0@g
    def fg(theta):
        logit=logrho+g@theta; logz=logsumexp(logit)
        p=np.exp(logit-logz)
        return logz-theta@moment,p@g-moment
    opt=minimize(lambda t: fg(t)[0],np.zeros(k),jac=lambda t:fg(t)[1],method='L-BFGS-B',
                 bounds=[(-80,80)]*k,options={'maxiter':180,'ftol':1e-12})
    theta=opt.x.copy()
    def pdf(th):
        lt=logrho+g@th
        return np.exp(lt-logsumexp(lt))
    def f(th):
        p=pdf(th);mu=p@g
        gc=g-mu
        C=gc.T@(gc*p[:,None]);C.flat[::k+1]+=regularize
        v=p@lg
        try:return np.linalg.solve(C,v)
        except np.linalg.LinAlgError:return np.linalg.lstsq(C,v,rcond=1e-9)[0]
    preds=[]
    for tprev,t in zip([0.]+list(times[:-1]),times):
        if t>tprev:
            steps=int(np.ceil((t-tprev)/max_step));dt=(t-tprev)/steps
            for _ in range(steps):
                v=np.clip(f(theta),-80,80)
                th2=theta+dt*v
                v2=np.clip(f(th2),-80,80)
                theta+=.5*dt*(v+v2)
                theta=np.clip(theta,-80,80)
        preds.append(pdf(theta))
    return np.array(preds),time.perf_counter()-t0,opt.success

def coherent_anchors(params,initial_thetas,T):
    # Macro-branch trajectories: dynamics of binomial-coherent parameters,
    # with reaction propensities evaluated under each product binomial.
    d=params['d'];L=params['L'];M=L-1;n=np.arange(L,dtype=float)
    H=hs(n,M/2,params['Hill_power']);cap=(M-n)/M
    def f(t,z):
        z=np.clip(z.reshape(initial_thetas.shape),1e-5,1-1e-5)
        res=np.zeros_like(z)
        for j in range(z.shape[0]):
            pp=np.array([comb(M,n)*x**n*(1-x)**(M-n) for x in z[j]])
            for i in range(d):
                prev=(i-1)%d
                hn=pp[prev]@H
                if not params['positive'][i]:hn=1-hn
                bi=pp[i]@(cap*(params['b0'][i]+params['auto'][i]*H+params['neighbor'][i]*hn))
                de=params['death'][i]*M*z[j,i]
                res[j,i]=(bi-de)/M
        return res.ravel()
    sol=solve_ivp(f,(0,T),initial_thetas.ravel(),t_eval=[0,T/2,T],rtol=1e-8,atol=1e-10)
    return np.clip(sol.y.T.reshape(3,*initial_thetas.shape).reshape(-1,d),.015,.985)

def multi_reference_basis(params,thetas,T,K,limit=None):
    d=params['d'];L=params['L'];N=L**d
    anchors=coherent_anchors(params,thetas,T)
    inds=total_degree_indices(d,L,K,max_support=None)
    pieces=[]
    for th in anchors:
        local=[orthogonal_local(L,float(t)) for t in th]
        for alpha in inds:
            col=np.array([1.])
            for i,q in enumerate(alpha):col=np.multiply.outer(col,local[i][:,q]).ravel()
            pieces.append(col)
    Braw=np.array(pieces).T
    gram=Braw.T@Braw
    lam,V=np.linalg.eigh(gram)
    retain=lam>max(lam[-1]*1e-9,1e-10)
    # The Gram spectrum determines the true number of independent modes.
    B=(Braw@V[:,retain])/np.sqrt(lam[retain])[None,:]
    if limit is not None and B.shape[1]>limit:
        # largest Gram eigenmodes capture the most coherent density mixture
        B=B[:,-limit:]
    return np.asfortranarray(B),len(anchors)*len(inds),len(anchors)

def binomial_product_from_theta(L,theta):
    M=L-1;n=np.arange(L,dtype=float)
    col=np.array([1.])
    for t in theta:
        q=comb(M,n)*t**n*(1-t)**(M-n)
        col=np.multiply.outer(col,q).ravel()
    return col/max(col.sum(),1e-300)

def coherent_ctmc(Q,params,initial_thetas,initial_weights,T,times,extra=0,delta=.12,
                  ridge=1e-8):
    # Positive coherent-state mixture with an autonomous reduced CTMC generator.
    # The nonnegative jump rates are least-squares fits to Q*rho_j, constrained
    # so that K is a Markov generator; after precomputation, propagate only w(t).
    # Precomputing Q*B and B.T B still costs O(full_states * mode_count^2).
    t0=time.perf_counter();d=params['d'];L=params['L']
    anchors=coherent_anchors(params,initial_thetas,T)
    # anchor ordering: 0..j-1 initially, j..2j-1 mid, remainder final
    atoms=[];links=[]
    for j,th in enumerate(anchors):
        atoms.append(binomial_product_from_theta(L,th)); links.append((j,'main'))
        if extra>=1:
            for i in range(d):
                tt=th.copy();tt[i]=np.clip(tt[i]+delta,.01,.99)
                atoms.append(binomial_product_from_theta(L,tt));links.append((j,'+%d'%i))
        if extra>=2:
            for i in range(d):
                tt=th.copy();tt[i]=np.clip(tt[i]-delta,.01,.99)
                atoms.append(binomial_product_from_theta(L,tt));links.append((j,'-%d'%i))
    B=np.stack(atoms,axis=1);m=B.shape[1]
    gram=B.T@B
    project=B.T@(Q@B)
    K=np.zeros((m,m))
    fits=[]
    for j in range(m):
        keep=np.arange(m)!=j
        ii=np.where(keep)[0]
        H=gram[np.ix_(ii,ii)]-gram[ii,j][:,None]-gram[j,ii][None,:]+gram[j,j]
        H.flat[::len(ii)+1]+=ridge
        f=project[ii,j]-project[j,j]
        def obj(a):return .5*a@H@a-f@a
        def jac(a):return H@a-f
        res=minimize(obj,np.zeros(len(ii)),jac=jac,method='L-BFGS-B',bounds=[(0,15)]*len(ii),
                     options={'maxiter':90,'ftol':1e-10})
        a=np.clip(res.x,0,None)
        K[ii,j]=a;K[j,j]=-a.sum()
        fits.append(float(np.linalg.norm((Q@B[:,j])-B@K[:,j])))
    weights0=np.zeros(m)
    j0=initial_thetas.shape[0]
    # first j0 anchor columns correspond exactly to product-mixture components.
    stride=1+extra*d
    for j in range(j0):weights0[j*stride]=initial_weights[j]
    weight_series=expm_multiply(K,weights0,start=0,stop=times[-1],num=len(times),endpoint=True)
    pred=weight_series@B.T
    pred=np.maximum(pred,0);pred/=pred.sum(axis=1)[:,None]
    elapsed=time.perf_counter()-t0
    return pred,elapsed,m,float(np.mean(fits)),float(max(abs(K.sum(axis=0))))

def predict_memory(Q,p0,B,low_count,ranks,times):
    # Exact first-shell augmented Mori-Zwanzig embedding, truncated by SVD
    # A_21 coupling. The auxiliary ODE is a finite-dimensional memory surrogate.
    A=B.T@(Q@B); n=A.shape[0]; k=low_count
    A_ll=A[:k,:k]; A_lh=A[:k,k:]; A_hl=A[k:,:k]; A_hh=A[k:,k:]
    U,s,Vt=np.linalg.svd(A_hl,full_matrices=False)
    c0=B.T@p0
    rows=[]
    for r in ranks:
        if r>U.shape[1]:continue
        W=U[:,:r]
        Ar=np.block([[A_ll,A_lh@W],[W.T@A_hl,W.T@A_hh@W]])
        x0=np.r_[c0[:k],W.T@c0[k:]]
        sol=expm_multiply(Ar,x0,start=0,stop=times[-1],num=len(times),endpoint=True)
        C=np.column_stack([sol[:,:k],sol[:,k:]@W.T])
        raw=C@B.T
        pred=np.maximum(raw,0);pred/=np.maximum(pred.sum(axis=1,keepdims=True),1e-300)
        rows.append((r,pred,float(np.maximum(-raw[1:],0).sum(axis=1).mean())))
    return rows

def build_LR(terms,d,L):
    spl=d//2
    result=[]
    for lcs in terms:
        A=lcs[0]
        for i in range(1,spl): A=kron(A,lcs[i],format='csr')
        B=lcs[spl]
        for i in range(spl+1,d): B=kron(B,lcs[i],format='csr')
        result.append((A,B))
    return result

def dynamic_low_rank(terms,p0,d,L,times,rank=4,dtmax=.02):
    """Matrix projector-splitting dynamical low rank; no full Q or truth used.
    Operates only on U,S,V and separable reaction generators A_k P B_k^T.
    Initial distribution matrix is factored via SVD once.
    """
    t0=time.perf_counter();m=L**(d//2);n=L**(d-d//2)
    LR=build_LR(terms,d,L)
    P0=p0.reshape(m,n)
    u,s,vt=svd(P0,full_matrices=False,lapack_driver='gesdd')
    r=min(rank,min(m,n))
    U=u[:,:r];V=vt[:r].T;S=np.diag(s[:r]);
    # Operators applied directly to reduced factors; never form full P in evolution.
    def fK(K,V,Cr=None):
        if Cr is None:Cr=[V.T@(R.T@V) for _,R in LR]
        out=np.zeros_like(K)
        for (A,R),C in zip(LR,Cr):out+=A@K@C
        return out
    def fS(S,U,V,Cu=None,Cv=None):
        if Cu is None:Cu=[U.T@(A@U) for A,_ in LR]
        if Cv is None:Cv=[V.T@(R.T@V) for _,R in LR]
        out=np.zeros_like(S)
        for a,b in zip(Cu,Cv):out-=a@S@b
        return out
    def fL(Lmat,U,Cu=None):
        if Cu is None:Cu=[U.T@(A@U) for A,_ in LR]
        out=np.zeros_like(Lmat)
        for (A,R),a in zip(LR,Cu):out+=R@Lmat@a.T
        return out
    def step(U,S,V,h):
        Cv=[V.T@(R.T@V) for _,R in LR]
        K=U@S; k1=fK(K,V,Cv);k2=fK(K+.5*h*k1,V,Cv)
        K=K+h*k2
        U1,R1=np.linalg.qr(K,mode='reduced')
        Cu=[U1.T@(A@U1) for A,_ in LR]
        k1=fS(R1,U1,V,Cu,Cv);k2=fS(R1+.5*h*k1,U1,V,Cu,Cv)
        S2=R1+h*k2
        Lmat=V@S2.T
        k1=fL(Lmat,U1,Cu);k2=fL(Lmat+.5*h*k1,U1,Cu)
        Lmat=Lmat+h*k2
        V1,R2=np.linalg.qr(Lmat,mode='reduced')
        Snew=R2.T
        return U1,Snew,V1
    preds=[];rawnegs=[]
    for a,b in zip([0.]+list(times[:-1]),times):
        steps=int(np.ceil((b-a)/dtmax)) if b>a else 0
        for k in range(steps):U,S,V=step(U,S,V,(b-a)/steps)
        pred=(U@S)@V.T
        rawnegs.append(float(np.maximum(-pred,0).sum()))
        pred=np.maximum(pred,0).ravel()
        pred/=max(pred.sum(),1e-300)
        preds.append(pred)
    return np.array(preds),time.perf_counter()-t0,rawnegs

def score(truth,pred,coords,L):
    # metrics at t>0, no oracle post-fit, p0 reported separately
    l1=np.sum(np.abs(truth-pred),axis=1)
    hell=np.sqrt(.5*np.sum((np.sqrt(np.maximum(truth,0))-np.sqrt(np.maximum(pred,0)))**2,axis=1))
    # event: simultaneous majority of first 2 species above midpoint
    event=(coords[0]>=int(np.ceil((L-1)/2))) & (coords[1]>=int(np.ceil((L-1)/2)))
    pr_true=truth[:,event].sum(axis=1);pr_pred=pred[:,event].sum(axis=1)
    eventerr=np.abs(pr_true-pr_pred)
    x=coords[0].astype(float);y=coords[1].astype(float)
    joint_truth=(truth*x[None,:]*y[None,:]).sum(axis=1)
    joint_pred=(pred*x[None,:]*y[None,:]).sum(axis=1)
    # normalized observable-scale error for first species pair
    correrr=np.abs(joint_true if False else joint_truth-joint_pred)/((L-1)**2)
    return dict(L1=float(l1[1:].mean()),L1_final=float(l1[-1]),L1_t0=float(l1[0]),
                Hellinger=float(hell[1:].mean()),event_abs=float(eventerr[1:].mean()),
                event_abs_final=float(eventerr[-1]),moment_pair_err=float(correrr[1:].mean()),
                TV=float(l1[1:].mean()/2))

def run_case(case,configs,quick=False):
    d,L,scenario=case
    caseid=f'{scenario}_d{d}_L{L}'
    print('\n###',caseid,flush=True)
    Q,terms,coords,params,T=build_network(d,L,scenario)
    check_Q(Q)
    p0,thetas,w=initial_p(d,L,scenario,coords)
    times=np.linspace(0,T,5 if not quick else 3)
    truth,truthsec=expm_truth(Q,p0,times)
    print('reference',Q.shape,Q.nnz,'seconds',round(truthsec,2),'mass',np.max(abs(truth.sum(axis=1)-1)),flush=True)
    rows=[];saved={'times':times,'truth':truth}
    def add(name,dim,pred,elapsed,**meta):
        if not np.all(np.isfinite(pred)):
            print('BAD NONFINITE',name); return
        met=score(truth,pred,coords,L)
        row=dict(case=caseid,scenario=scenario,d=d,L=L,full_states=L**d,method=name,parameters=dim,seconds=elapsed,**met,**meta)
        rows.append(row)
        print('%-33s %5d L1 %.4f (init %.4f) evt %.4f sec %.2f'%(name,dim,met['L1'],met['L1_t0'],met['event_abs'],elapsed),flush=True)
        saved[name]=pred
        pd.DataFrame(rows).to_csv(os.path.join(OUT,f'{caseid}_scores_partial.csv'),index=False)
    add('exact_CME_Krylov',L**d,truth,truthsec,valid_probability=True,scalable=False)
    a,sec=predict_meanfield(Q,coords,params,thetas,w,times)
    add('mean_field_or_mixture',len(thetas)*d*L+len(thetas)-1,a,sec,valid_probability=True,scalable=True)
    # Fock projected models, bases entirely specified by rates and initial distribution,
    # no future information or truth used.
    base=orthogonal_local(L,.47)
    maxK=configs['degrees']
    for degree in maxK:
        B,inds=get_basis(d,L,degree,local_basis=base)
        n=B.shape[1]
        print(' basis degree',degree,'n',n,flush=True)
        if n>=configs.get('max_columns',800):continue
        # linear signed projection not always a valid probability: reveal negative mass
        try:
            pred,raw,sec=predict_linear(Q,p0,B,times)
            neg=float(np.maximum(-raw[1:],0).sum(axis=1).mean())
            add('linear_Fock_degree_'+str(degree),n,pred,sec,raw_negative=neg,valid_probability=False,scalable=False)
        except Exception as ex:print('linear failed',degree,ex,flush=True)
        try:
            pred,sec=predict_amp(Q,p0,B,times,eps_factor=1e-5)
            add('square_amplitude_Fock_degree_'+str(degree),n,pred,sec,valid_probability=True,scalable=False)
        except Exception as ex:print('amp failed',degree,ex,flush=True)
        if degree<=configs.get('max_entropy_degree',2) and n<=configs.get('max_entropy_modes',28):
            try:
                pred,sec,initok=predict_entropy(Q,p0,B,times)
                add('entropy_Fock_degree_'+str(degree),n-1,pred,sec,valid_probability=True,scalable=False,entropy_init_success=initok)
            except Exception as ex:print('entropy failed',degree,ex,flush=True)
        del B
    # Multiple coherent references anchored only to initial macro-regimes and
    # their cheap mean-field evolution, never to the reference distribution.
    for degree in configs.get('multi_degrees',[]):
        try:
            B,raw_k,nref=multi_reference_basis(params,thetas,T,degree,limit=configs.get('multi_max_columns',240))
            print('multi coherent degree',degree,'independent',B.shape[1], 'raw',raw_k,flush=True)
            pred,sec=predict_amp(Q,p0,B,times,eps_factor=1e-5)
            add('multi_coherent_amp_degree_'+str(degree),B.shape[1],pred,sec,
                valid_probability=True,scalable=False,coherent_anchors=nref)
            del B
        except Exception as ex:print('multi coherent failed',degree,repr(ex),flush=True)
    for extra in configs.get('coherent_extra',[]):
        try:
            pred,sec,dim,resid,massres=coherent_ctmc(Q,params,thetas,w,T,times,extra=extra)
            add('coherent_Markov_Fock_extra_'+str(extra),dim,pred,sec,
                markov_residual=resid,reduced_mass_residual=massres,
                valid_probability=True,scalable='after_full_state_preprocessing')
        except Exception as ex:print('coherent Markov failed',extra,repr(ex),flush=True)
    if configs.get('memory',False):
        try:
            Bbig,inds=get_basis(d,L,3,local_basis=base)
            low_n=len(total_degree_indices(d,L,2))
            tt=time.perf_counter()
            for r,pr,neg in predict_memory(Q,p0,Bbig,low_n,[2,4,8,16],times):
                add('Fock_MZ_aux_'+str(r),low_n+r,pr,time.perf_counter()-tt,
                    valid_probability=False,scalable=False,raw_negative=neg)
        except Exception as ex:print('memory failed',repr(ex),flush=True)
    for r in configs['ranks']:
        try:
            pred,sec,negs=dynamic_low_rank(terms,p0,d,L,times,rank=r,dtmax=.025)
            m=L**(d//2);n=L**(d-d//2)
            paramsr=r*(m+n-r)
            add('matrix_DLR_rank_'+str(r),paramsr,pred,sec,raw_negative=np.mean(negs[1:]),valid_probability=False,scalable=True)
        except Exception as ex: print('DLR failed',r,ex,flush=True)
    # save only small subset distributions, no entire exact if 279k * 5 to zip ? yes ~11MB
    np.savez_compressed(os.path.join(OUT,f'{caseid}_reference.npz'),times=times,truth=truth,
                        p0=p0,coords=coords)
    with open(os.path.join(OUT,f'{caseid}_parameters.json'),'w') as f:json.dump(params,f,indent=2)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--quick',action='store_true')
    p.add_argument('--case',default='all')
    args=p.parse_args()
    cases=[(4,7,'independent'),(4,7,'weak'),(5,6,'bistable'),(6,6,'cascade'),(7,6,'stiff')]
    if args.quick:cases=cases[:2]
    if args.case!='all':cases=[a for a in cases if str(a[0])==args.case or a[2]==args.case]
    allrows=[]
    for d,L,scenario in cases:
        conf=dict(degrees=([1,2,3,4,5] if d<=5 else [1,2,3,4] if d==6 else [1,2,3]),
                  ranks=([1,2,4,8] if d<=6 else [2,4,8]),max_columns=500,
                  max_entropy_modes=25,max_entropy_degree=2,
                  multi_degrees=([0,1,2,3] if d<=5 else [0,1,2] if d==6 else [0,1]),
                  multi_max_columns=(240 if d<=6 else 75),memory=(d<=6),
                  coherent_extra=([0,1,2] if d<=5 else [0,1] if d==6 else [0]))
        allrows+=run_case((d,L,scenario),conf,quick=args.quick)
        pd.DataFrame(allrows).to_csv(os.path.join(OUT,'all_scores.csv'),index=False)
    print('\nDONE',len(allrows),'comparisons',flush=True)
if __name__=='__main__':main()
