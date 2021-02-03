# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:03:18 2020
This code dealed with the prediction of Neural network DEM after training
Only need the initial particle position information
The full traning algorithm will be published later
@author: Xizhong.Chen@ucc.ie or suningchen@gmail.com
"""

import numpy as np
import os
import argparse
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from scipy.integrate import odeint
from utis.plotting import Out_vtk, plot_par, pre_data, post_data, makedirs
import nodepy.linear_multistep_method as lm 
import pandas as pd
# Force Float 64
tf.keras.backend.set_floatx('float64')
plt.style.use('seaborn-paper')
print('tensor flow verision',tf.__version__)

#%%global parameters
device = 'cpu:0' # These experiments do not require the GPU. Normally, 'gpu:0' 
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weights", type=eval, default=True)
ckpt = 'model_weights/ckpt_verlet'
args = parser.parse_args()
#%% hyper parameters
batch_size = 5000 #also have repeat and skip function, reshuffle_each_iteration=True
batch_buffer = 1# this is a chunk of batch, to shuffle, 1 is not shuffle, also window
epochs =1000
initial_learning_rate = 0.005 #use smaller
log_freq = 25
decay_steps = 250
decay_rate = 0.9 #lr0 * decay_rate ^ (step / decay_steps)
lbfgs_op = False # using lbfgs optimization
#%%Load DEM data
inputFile = './dataset/rpm2_4s.mat'
Radius = 2.0e-3;
scheme = 'AM' #integrators
M = 1
lw_nt = 1  #number of recursive loss step
lw_t = np.zeros(lw_nt); lw_t[-1] =1. #weight of each recursive loss
nts = 0.25; nte=1.0; ndof=3; scale=0
out_vtk= 1
noise = 0.0

Pid, X_train,t_train, Scale= pre_data(
    inputFile, nts=nts, nte=nte, ndof=ndof, scale=scale)
#X_train = X_train[:4,:2,:] ; t_train=t_train[:4]
nt  = X_train.shape[0]; npar  = X_train.shape[1];  nd  = X_train.shape[2];

X_train = X_train.reshape(nt,-1)
X_train = X_train + np.random.normal(0, noise, X_train.shape)
y0 = X_train[0,:]; Y_train = X_train;  dt = t_train[1] - t_train[0]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=batch_buffer).batch(batch_size)    
#%%MOC trainable NN  
class VNN(tf.keras.Model):  
    def __init__(self, X, lw_t, nd, dt=0.1, M=1, scheme='AM', **kwargs):
        super().__init__(**kwargs)
        
        super().__init__(**kwargs)
        
        self.dt = dt
        self.X = X # S x N x D
        
        self.nt = X.shape[0] # number of time snapshots is nt
        self.nd = nd
        self.npd = X.shape[1] # np* nd  
        self.scheme = scheme
        self.lw_t = lw_t * self.npd
        npd = self.npd        
        
        #This is simply NN not NODE
        self.neural_net = tf.keras.Sequential([
                 tf.keras.layers.Dense(80, activation=tf.nn.tanh, input_shape=(npd,) ),
                 tf.keras.layers.Dense(80, activation=tf.nn.tanh, kernel_initializer='glorot_normal'),                 
                 tf.keras.layers.Dense(80, activation=tf.nn.tanh, kernel_initializer='glorot_normal'),
                 tf.keras.layers.Dense(npd, activation=None,name='output'),   ##we can change the NN here
        ])
        self.neural_net.summary()
        # Load weights
        self.M = M # number of Adams-Moulton steps
        switch = {'AM': lm.Adams_Moulton}
        method = switch[scheme](M)
        self.alpha = np.float32(-method.alpha[::-1]) 
        self.beta = np.float32(method.beta[::-1])
    def call(self, X): #The full algorithm to be pulished soon
        M = self.M

    def learned_f(self,x,t=None):
        f = self.neural_net(x[None,:]) #input x grids
        return f.numpy().flatten() # predict dNdt             
    
#%%Adam
start = datetime.now()
with tf.device(device):
    model = VNN(X_train, lw_t, nd, dt, M, scheme)  
    print("Ground truth shape :", X_train.shape)
    
    if args.weights:
        print('Loading previous save models')
        model.load_weights(ckpt)
    else: #train the modelling   
        Nit = []; trainLoss=[]; valLoss=[]
        #The full algorithm to be pulished soon
#%%bfgs    
print("Finished Training ! ", "Duration time:", datetime.now()-start) 
#%%prediction   
makedirs('png')
Pid, X_train, t_train, Scale= pre_data(
    inputFile, nts=nts, nte=nte, ndof=ndof, scale=scale)
X_train = X_train.reshape(nt,-1); 
t_star = t_train; X_star = X_train

y0 = X_star[0,:]
learned_X = odeint(model.learned_f, y0, t_star)


x_star = np.zeros_like(X_star.reshape(nd,-1,npar) ) 
pred_out = np.zeros_like(learned_X.reshape(nd,-1,npar) )   
for i in range(0,nd):
    pred_out[i,:,:] = learned_X.reshape(-1,npar,nd)[:,:,i]
    x_star[i,:,:] = X_star.reshape(-1,npar,nd)[:,:,i]
    plot_par(Pid,t_star, pred_out[i,:,:], x_star[i,:,:], i)
    plt.figure()
    plt.plot(t_star, pred_out[i,:,0]); plt.plot(t_star, x_star[i,:,0])
    plt.savefig('./p'+str(i)+'.png',bbox_inches='tight',dpi=150);  plt.show(block=False)   

    