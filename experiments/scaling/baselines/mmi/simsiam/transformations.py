import numpy as np
from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat
import tensorflow as tf


def DA_Jitter(X, sigma=0.8):
    """Add Gaussian noise"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape).astype(np.float32)
    return (X + myNoise).astype(np.float32)


def DA_Scaling(X, sigma=0.1):
    """Random scaling"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1])).astype(np.float32)
    return (X * scalingFactor).astype(np.float32)


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    """Generate random curves for magnitude warping"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1])).astype(np.float32)
    x_range = np.arange(X.shape[0])
    
    cs = []
    for i in range(X.shape[1]):
        cs.append(CubicSpline(xx[:, i], yy[:, i])(x_range))
    
    ret_lst = np.array(cs, dtype=np.float32).transpose()
    return ret_lst


def DA_MagWarp(X, sigma=0.3):
    """Magnitude warping"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    result = X * GenerateRandomCurves(X, sigma)
    return result.astype(np.float32)


def DistortTimesteps(X, sigma=0.2):
    """Distort timesteps for time warping"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    tt = GenerateRandomCurves(X, sigma)
    tt_cum = np.cumsum(tt, axis=0)
    
    # Scale to match original length
    t_scale = (X.shape[0] - 1) / tt_cum[-1, :]
    tt_cum = tt_cum * t_scale
    
    return tt_cum


def DA_TimeWarp(X, sigma=0.2):
    """Time warping"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape, dtype=np.float32)
    x_range = np.arange(X.shape[0])
    
    for i in range(X.shape[1]):
        X_new[:, i] = np.interp(x_range, tt_new[:, i], X[:, i])
    
    return X_new


def DA_Rotation(X, sigma=0):
    """Rotation augmentation"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X, axangle2mat(axis, angle)).astype(np.float32)


def DA_Permutation(X, nPerm=4, minSegLength=10, sigma=0):
    """Permutation augmentation"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    X_new = np.zeros(X.shape, dtype=np.float32)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
        X_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return X_new


def RandSampleTimesteps(X, nSample=1000):
    """Random sample timesteps"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    tt = np.zeros((nSample, X.shape[1]), dtype=int)
    for i in range(X.shape[1]):
        tt[1:-1, i] = np.sort(np.random.randint(1, X.shape[0] - 1, nSample - 2))
    tt[-1, :] = X.shape[0] - 1
    return tt


def DA_RandSampling(X, nSample=1000, sigma=0):
    """Random sampling augmentation"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros(X.shape, dtype=np.float32)
    for i in range(X.shape[1]):
        X_new[:, i] = np.interp(np.arange(X.shape[0]), tt[:, i], X[tt[:, i], i])
    return X_new


def DA_Combined(X, nPerm=4, minSegLength=10, sigma=0):
    """Combined augmentation"""
    X_new = DA_Permutation(X)
    return DA_Rotation(X_new)


def DA_Negation(X, sigma=0):
    """Negation augmentation"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    return (-1 * X).astype(np.float32)


def DA_Flip(X, sigma=0):
    """Flip augmentation"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    return np.flip(X, 0).copy().astype(np.float32)


def DA_ChannelShuffle(X, sigma=0):
    """Channel shuffle augmentation"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    indx = np.arange(X.shape[1])
    np.random.shuffle(indx)
    return X[:, indx].astype(np.float32)


def DA_Drop(X, W=7, sigma=10):
    """Drop augmentation"""
    # Ensure X is numpy array
    if hasattr(X, 'numpy'):
        X = X.numpy()
    W = sigma
    X_new = X.copy()
    indx = np.arange(X.shape[0] - W)
    np.random.shuffle(indx)
    X_new[indx[0]:indx[0] + W, :] = 0
    return X_new.astype(np.float32)
