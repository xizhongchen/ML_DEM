#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import os, shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax
def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def plot_par(x,t_star, learned_X_star, True_X_star, i):
        
    ####### Plotting ##################    
    plt.figure()
    plt.plot(t_star,True_X_star[:,0],'r-')
    plt.plot(t_star,learned_X_star[:,0],'k--')
    plt.savefig('./png/p1'+str(i)+'.png', bbox_inches='tight',dpi=150)
    #plt.show()

    fig, ax = newfig(1.0, 1.55)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(3, 2)
    gs0.update(top=0.95, bottom=0.35, left=0.1, right=1.0, hspace=0.5, wspace=0.3)
    
    ax = plt.subplot(gs0[0:1, 0:1])
    ax.plot(t_star,True_X_star[:,0],'r-', label='actual')
    ax.plot(t_star,learned_X_star[:,0],'k--', label='learned')
    ax.set_xlabel('t')
    ax.set_ylabel('S_1')
    
    ax = plt.subplot(gs0[0:1, 1:2])
    ax.plot(t_star,True_X_star[:,1],'r-')
    ax.plot(t_star,learned_X_star[:,1],'k--')
    ax.set_xlabel('t')
    ax.set_ylabel('S_2')

    ax = plt.subplot(gs0[1:2, 0:1])
    ax.plot(t_star,True_X_star[:,2],'r-')
    ax.plot(t_star,learned_X_star[:,2],'k--')
    ax.set_xlabel('t')
    ax.set_ylabel('S_3')
    
    ax = plt.subplot(gs0[1:2, 1:2])
    ax.plot(t_star,True_X_star[:,3],'r-')
    ax.plot(t_star,learned_X_star[:,3],'k--')
    ax.set_xlabel('t')
    ax.set_ylabel('S_4')
    
    ax = plt.subplot(gs0[2:3, 0:1])
    ax.plot(t_star,True_X_star[:,4],'r-')
    ax.plot(t_star,learned_X_star[:,4],'k--')
    ax.set_xlabel('t')
    ax.set_ylabel('S_5')
    
    ax = plt.subplot(gs0[2:3, 1:2])
    ax.plot(t_star,True_X_star[:,5],'r-')
    ax.plot(t_star,learned_X_star[:,5],'k--')
    ax.set_xlabel('t')
    ax.set_ylabel('S_6')
   
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=False)
    plt.savefig('./png/p'+str(i)+'.png',bbox_inches='tight',dpi=150)    
    #plt.show()    

def output_vtk(saveName, idd, PX, PY, PZ, Radius=2.0e-3, Vel=None):
    npar = len(PX)
    file = open(saveName,"w")
    file.write('# vtk DataFile Version 2.0\n')
    file.write('Particle data\nASCII\n')
    file.write('DATASET POLYDATA\n')
    file.write('POINTS {} double\n'.format(npar))
    for i in range(npar):
        file.write('{:.8e}\t{:.8e}\t{:.8e}\n'.format(PX[i],PY[i],PZ[i])) 
    file.write( 'VERTICES {} {}\n'.format(npar, npar*2) )
    for i in range(npar):
        file.write('1 {}\n'.format(idd[i]) )
    file.write( 'POINT_DATA {}\n'.format(npar))
    file.write( 'SCALARS Radius double 1\n')
    file.write( 'LOOKUP_TABLE default\n')
    for i in range(npar):
        file.write('{:.8e}\n'.format(Radius))
    if Vel is not None:
        file.write( 'VECTORS velocity double\n')
        for i in range(npar):
            file.write('{:.8e}\t{:.8e}\t{:.8e}\n'.format(Vel[0,i],Vel[1,i],Vel[2,i]))         
    file.close()

def pre_data(inputFile, nts=0.5, nte=0.9, ndof=3, scale=0):
    import scipy.io
    from sklearn import preprocessing
    
    data = scipy.io.loadmat(inputFile)
    
    T = data['t'].flatten()#[:,None] # T x 1
    Pid = data['id'].flatten() # N x 1 # N x 1
    PX = np.real(data['PX']).T # T x N 
    PY = np.real(data['PY']).T # T x N 
    PZ = np.real(data['PZ']).T # T x N
    if ndof == 6:
        VX = np.real(data['VX']).T # T x N 
        VY = np.real(data['VY']).T # T x N 
        VZ = np.real(data['VZ']).T # T x N        
    #normalize each particle position x, y, z
    nt = len(T); npar =len(Pid)
    Scale = []
    for i in range(ndof):        
        if scale == 0:
            Scale.append(0)
        else:
            if scale == 1:
                Scale.append(preprocessing.MinMaxScaler())
                ScaleX = preprocessing.MinMaxScaler()           
            elif scale == 2:
                Scale.append(preprocessing.MaxAbsScaler())            
            else:
                Scale.append(preprocessing.StandardScaler())
               
    if scale:
        PX = np.reshape(Scale[0].fit_transform(np.reshape(PX,(-1,1))) , (nt,npar) )
        PY = np.reshape(Scale[1].fit_transform(np.reshape(PY,(-1,1))) , (nt,npar) )
        PZ = np.reshape(Scale[2].fit_transform(np.reshape(PZ,(-1,1))) , (nt,npar) )
        if ndof == 6: 
             VX = np.reshape(Scale[3].fit_transform(np.reshape(VX,(-1,1))) , (nt,npar) )
             VY = np.reshape(Scale[4].fit_transform(np.reshape(VY,(-1,1))) , (nt,npar) )
             VZ = np.reshape(Scale[5].fit_transform(np.reshape(VZ,(-1,1))) , (nt,npar) ) 
             
    if ndof == 6:              
         PP = np.stack((PX, PY, PZ, VX, VY, VZ)).transpose(1,2,0) #(nt, np, nd)
    else:
         PP = np.stack((PX, PY, PZ)).transpose(1,2,0) #(nt, np, nd)                        
    
    ts = int(len(T)*nts) #only train the steady state
    te = int(len(T)*nte) # last 10 for test
    SamTr = slice(ts,te,1) ; 
    t_train = T[SamTr];   
    x_train =  PP[SamTr,:,:] #(nt, np, nd) 
    
    return Pid, x_train, t_train, Scale

def post_data(Scale, X):
    if Scale: #scale !=0
        nt = X.shape[0]; npar = X.shape[1];
        X_res = Scale.inverse_transform(np.reshape(X,(-1,1))).reshape(nt, npar)
    else:
        X_res = X
    return X_res

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def Out_vtk(dirname, Pid, Radius, pred_out, x_star):
    print(os.getcwd())        
    if  os.path.exists(dirname):
        shutil.rmtree(dirname, ignore_errors=True)
    os.makedirs(dirname)
    nt = pred_out.shape[1]
    nd = pred_out.shape[0]
    for i in range(nt):
        saveName = (f"./{dirname}/learn_{i:05d}.vtk")
        #PXi = learnX.T[:,i]; PYi = learnY.T[:,i]; PZi = learnZ.T[:,i]
        PXi = pred_out[0,i,:]; PYi = pred_out[1,i,:]; PZi = pred_out[2,i,:]
        if nd == 6:
            Vel = pred_out[3:,i,:]
        else:
            Vel =None
        output_vtk(saveName, Pid, PXi, PYi, PZi, Radius, Vel)
    for i in range(x_star.shape[1]):        
        saveName = (f"./{dirname}/True_{i:05d}.vtk")
        if nd == 6:
            Vel = x_star[3:,i,:]
        else:
            Vel =None        
        PXi = x_star[0,i,:]; PYi = x_star[1,i,:]; PZi = x_star[2,i,:]
        output_vtk(saveName, Pid, PXi, PYi, PZi, Radius, Vel)         
