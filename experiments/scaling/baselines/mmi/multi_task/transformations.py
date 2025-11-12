import numpy as np
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
import tensorflow as tf

def DA_Jitter(X, sigma=0.8):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def DA_Scaling(X, sigma=1.1*0.5):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs=[]
    for i in range(X.shape[1]):
        cs.append(CubicSpline(xx[:,i], yy[:,i])(x_range))
    ret_lst=np.array(cs).transpose()
    return ret_lst
    
def DA_MagWarp(X, sigma = 0.5):
    return X * GenerateRandomCurves(X, sigma)

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples around 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,i] for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        tt_cum[:,i] = tt_cum[:,i]*t_scale[i]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(X.shape[1]):
        X_new[:,i] = np.interp(x_range, tt_new[:,i], X[:,i])
    
    return X_new

def DA_Rotation(X,sigma=0):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))

def DA_Permutation(X, nPerm=4, minSegLength=10, sigma=0):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)

def RandSampleTimesteps(X, nSample):
    tt = np.zeros((nSample, X.shape[1]), dtype=int)
    
    for i in range(X.shape[1]):
        # Generate unique random sample points
        # Sample from interior points only, always include endpoints
        if nSample <= 2:
            tt[:, i] = [0, X.shape[0] - 1]
        else:
            # Sample nSample-2 random interior points
            interior_indices = np.sort(np.random.choice(
                np.arange(1, X.shape[0] - 1), 
                size=min(nSample - 2, X.shape[0] - 2), 
                replace=False
            ))
            tt[0, i] = 0
            tt[1:len(interior_indices)+1, i] = interior_indices
            tt[len(interior_indices)+1, i] = X.shape[0] - 1
            
            # If nSample is larger than we can fit, pad with endpoint
            if len(interior_indices) + 2 < nSample:
                tt[len(interior_indices)+2:, i] = X.shape[0] - 1
    
    return tt

def DA_RandSampling(X, nSample=None, sigma=0):
    # Use a fixed small number of samples for noticeable effect on short signals
    if nSample is None:
        nSample = 15  # Fixed value that creates noticeable distortion for 40-step signals
    
    # Ensure nSample is at least 3 and less than signal length
    nSample = max(3, min(nSample, X.shape[0] - 1))
    
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    
    # Generate random sample points for each channel independently
    for i in range(X.shape[1]):
        # Sample random indices without replacement, then sort
        sample_indices = np.sort(np.random.choice(
            X.shape[0], 
            size=nSample, 
            replace=False
        ))
        
        # Interpolate from sampled points back to full length
        # Linear interpolation smooths the signal between sample points
        X_new[:, i] = np.interp(x_range, sample_indices, X[sample_indices, i])
    
    return X_new

def DA_Combined(X, nPerm=4, minSegLength=10, sigma=0):
    X_new=DA_Permutation(X)
    return DA_Rotation(X_new)
    
def DA_Negation(X,sigma=0):
    return -1*X

def DA_Flip(X,sigma=0):
    return np.flip(X,0)

def DA_ChannelShuffle(X, sigma=0):
    indx=np.arange(X.shape[1])
    np.random.shuffle(indx)
    return X[:,indx]
    
def DA_Drop(X, W=7, sigma=0):
    """Drop a random window of the signal"""
    if sigma is not None and sigma != 0:
        W = int(sigma)
    else:
        W = int(W)
    
    # Ensure W is valid
    if W >= X.shape[0]:
        W = max(1, X.shape[0] - 1)
    if W <= 0:
        return X.copy()
    
    X_new = X.copy()  # Don't modify in place
    
    # Only drop if we have enough samples
    if X.shape[0] > W:
        indx = np.arange(X.shape[0] - W)
        if len(indx) > 0:
            np.random.shuffle(indx)
            X_new[indx[0]:indx[0]+W, :] = 0
    
    return X_new
