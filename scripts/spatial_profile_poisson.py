"""Cell-intercept-profiled Poisson likelihood and annual-score HAC covariance."""
import numpy as np
from scipy.optimize import minimize


def likelihood(beta,x,y,cell,weights=None):
    weights=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    eta=x@beta
    ncell=int(np.max(cell))+1
    shift=np.full(ncell,-np.inf)
    np.maximum.at(shift,cell,eta)
    centred=eta-shift[cell]
    exponential=np.exp(centred)*weights
    denom=np.bincount(cell,weights=exponential,minlength=ncell)
    totals=np.bincount(cell,weights=y*weights,minlength=ncell)
    multiplier=np.divide(totals,denom,out=np.zeros(ncell),where=denom>0)
    mu=exponential*multiplier[cell]
    active=totals>0
    value=float(-(y*weights*centred).sum()+(totals[active]*np.log(denom[active])).sum())
    gradient=x.T@(mu-y*weights)
    return value,gradient,mu,totals


def hac_covariance(bread,scores,maxlags=3):
    if maxlags<0 or maxlags>=len(scores):raise ValueError('Invalid annual HAC bandwidth')
    meat=scores.T@scores
    for lag in range(1,maxlags+1):
        cross=scores[lag:].T@scores[:-lag]
        meat+=(1-lag/(maxlags+1))*(cross+cross.T)
    return bread@meat@bread*len(scores)/(len(scores)-1)


def fit(x,y,cell,year,maxlags=3,start=None):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    cell=np.asarray(cell,dtype=int);year=np.asarray(year,dtype=int)
    total=y.sum()
    if total<=0:raise ValueError('No events available')
    def objective(beta):
        value,gradient,_,_=likelihood(beta,x,y,cell)
        return value/total,gradient/total
    solution=minimize(objective,np.zeros(x.shape[1]) if start is None else start,
        jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-13,'gtol':1e-8,'maxls':40})
    if not solution.success:raise RuntimeError(f'Profile Poisson optimization failed: {solution.message}')
    value,gradient,mu,totals=likelihood(solution.x,x,y,cell)
    ncell=len(totals)
    cell_x=np.stack([np.bincount(cell,weights=mu*x[:,k],minlength=ncell) for k in range(x.shape[1])],axis=1)
    divide=np.divide(cell_x,totals[:,None],out=np.zeros_like(cell_x),where=totals[:,None]>0)
    information=x.T@(mu[:,None]*x)-cell_x.T@divide
    eigenvalues=np.linalg.eigvalsh(information)
    if eigenvalues[0]<=1e-10:raise RuntimeError('Profile information matrix is singular')
    bread=np.linalg.inv(information)
    years=np.arange(year.min(),year.max()+1)
    # Efficient cluster scores must project out fitted cell intercepts.
    # Uncentred per-winter scores would omit nuisance-estimation uncertainty.
    efficient=x-divide[cell]
    scores=np.stack([efficient[year==wy].T@(y[year==wy]-mu[year==wy]) for wy in years])
    covariance=hac_covariance(bread,scores,maxlags)
    return {'params':solution.x,'covariance':covariance,'mu':mu,
        'information_condition_number':float(eigenvalues[-1]/eigenvalues[0]),
        'max_scaled_gradient':float(np.max(np.abs(gradient))/total),
        'iterations':int(solution.nit),'n_events':int(total),'n_rows':len(y),
        'n_informative_cells':int((totals>0).sum()),'n_winters':len(years),
        'profile_negative_log_likelihood':value,'information_inverse':bread,'annual_efficient_scores':scores}
