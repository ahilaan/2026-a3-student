"""Label-free query min-max transform; train using compliant_train.py."""
import numpy as np

def augment(X):
    """Retain raw values; append finite-value min-max scores within one query."""
    raw=np.asarray(X,dtype=np.float32)
    if raw.ndim!=2:raise ValueError('Expected a matrix')
    n,width=raw.shape;out=np.zeros((n,2*width),dtype=np.float32);out[:,:width]=raw
    if not n:return out
    values=raw.astype(np.float64);finite=np.isfinite(values)
    lo=np.min(np.where(finite,values,np.inf),axis=0)
    hi=np.max(np.where(finite,values,-np.inf),axis=0)
    span=hi-lo;active=np.isfinite(span)&(span>0)
    normalized=out[:,width:]
    normalized[:,active]=((values[:,active]-lo[active])/span[active]).astype(np.float32)
    normalized[~finite]=np.nan
    return out

