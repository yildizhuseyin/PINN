
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras

import matplotlib.pyplot as plt
from matplotlib import cm

import math
import numpy as np
import time as t

from sympy import *


def get_random_color():
    r=np.random.random([3,])
    rnd_color=(r[0], r[1], r[2])
    return rnd_color

def plot_list_points(figNo,subNo,plotList,title=None,figsize =(8, 7)):
    level=500;
    fig = plt.figure(figNo,figsize)
    ax = fig.add_subplot(subNo) #♦  projection='3d'
    
    for newData in plotList:
        x,y,color=newData
        if color==None:
            color=get_random_color()
        
        ax.plot(x, y,color)
    

    if not title==None:
        ax.set_xlabel('x')
        ax.set_ylabel('Y')
        ax.set_title(title)

def plot_points(figNo,subNo,X,Y,color=None,title=None,figsize =(8, 7)):
    level=500;
    fig = plt.figure(figNo,figsize)
    ax = fig.add_subplot(subNo) #♦  projection='3d'
    if color==None:
        ax.plot(X, Y)
    else: 
        ax.plot(X, Y,color)
    if not title==None:
        ax.set_xlabel('x')
        ax.set_ylabel('Y')
        ax.set_title(title)  
        
def plot_surf_3D(figNo,subNo,X,Y,Z,title):
    # bir veri grubu için yüzey çizimini yapar 
    level=30;
    fig = plt.figure(figNo,figsize =(8, 7))
    ax = fig.add_subplot(subNo,projection='3d') #♦  projection='3d' 
    levels = tf.linspace(-1, 1, level)
    # surf=ax.contourf(X, Y, Z, rstride=1, cstride=1, cmap='autumn',
    #     linewidth=0, antialiased=False)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
def plot_surf_2D(figNo,subNo,X,Y,Z,title):
    # bir veri grubu için yüzey çizimini yapar 
    level=50;
    fig = plt.figure(figNo,figsize =(8, 7))
    ax = fig.add_subplot(subNo) #♦  projection='3d' 
    levels = tf.linspace(-1, 1, level)
    surf=ax.contourf(X, Y, Z, cmap='autumn', antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

        
def tf_linsolve(M,B):
        B=tf.reshape(B,[-1,1])
        s1,s2=M.shape
        if s1>s2:
            Mt = tf.transpose(M);
            MM=tf.linalg.matmul(Mt,M)
            BB=tf.linalg.matmul(Mt,B);
        elif s2>s1:
            Mt = tf.linalg.transpose(M);
            #Bt = tf.linalg.transpose(B)
            MM=np.matmul(Mt,M);
            BB=np.dot(B,M);
        else:
            MM=M
            BB=B
        dC=tf.linalg.solve(MM,BB)  
        return tf.reshape(dC,[-1,1]) 
    
    
class normalizer_for_heat_problem:
    def __init__(self,T0,X0,k0): 
        self.T0=T0
        self.X0=X0
        self.k0=k0
    def normalize(self,var,par):
        if par=='X':
            nVar=var/self.X0
        elif par=='T':
            nVar=var/self.X0
        elif par=='k':
            nVar=var/self.k0
        elif par=='q':
            nVar=var/self.k0
            
            
 
class points_1D:
    def __init__(self,fcn=None):

        self.fcn=fcn
        self.np_x=[]
        self.np_f=[]
        self.np_k=[]
        self.np_dk=[]
        self.np_q=[]
        self.point_count=0
        self.isTensor=False
        
    def add_point(self,x,f,k,dk,q):
        self.np_x.append(x)
        self.np_f.append(f)
        self.np_k.append(k)
        self.np_dk.append(dk)
        self.np_q.append(q)
        self.point_count+=1
    def convert_tensor(self,dType):
        self.X=tf.reshape(tf.Variable(self.np_x,dtype=dType),[-1,1])
        self.F=tf.reshape(tf.convert_to_tensor(self.np_f,dtype=dType),[-1,1])
        self.k=tf.reshape(tf.convert_to_tensor(self.np_k,dtype=dType),[-1,1])
        self.dk=tf.reshape(tf.convert_to_tensor(self.np_dk,dtype=dType),[-1,1])
        self.q=tf.reshape(tf.convert_to_tensor(self.np_q,dtype=dType),[-1,1])
        self.isTensor=True
        print(self.dk)
        
        
        
class GEOMETRY_1D:
    def __init__(self,x_min,x_max,par,q0=0,n=10): 
        self.x_min=x_min; # başlangıç değeri, 
        self.x_max=x_max; # son değer, 
        self.n=n; # nokta sayısı 
        self.X=np.linspace(self.x_min,self.x_max,self.n)
        self.F=np.zeros_like(self.X)
        self.par=np.ones_like(self.X)*par
        self.q=np.ones_like(self.X)*q0
    def set_function_value(self,new_value,condition,Type='f'):#lambda x: x==0    # f: func value p: parameter value 
        # koşul yazılırken değişken x olarak girilmeli 
        for i in range(len(self.X)):
            x=self.X[i]
            if condition(x):
                print('nokta bulundu',x)
                if Type=='f':
                    self.F[i]=(new_value)
                elif Type=='q':
                    self.q[i]=new_value
                elif Type=='p':
                    self.par[i]=new_value
                else: 
                    print('Hatalı işlem yaptınız')
    def get_value(self,ids): 
        return self.X[ids],self.F[ids],self.par[ids],self.Dpar[ids],self.q[ids]
    
    def get_derivative_of_par(self):
        self.derivative_of_parameters=np.zeros_like(self.X)
        self.Dx=np.zeros_like(self.X)
        self.Dx[1:-1]=self.X[2:]-self.X[1:-1]
        self.Dx[0]=self.X[1]-self.X[0]
        self.Dx[-1]=self.X[-1]-self.X[-2]
        
        self.Dpar=np.zeros_like(self.X)
        self.Dpar[1:-1]=0.5*(self.par[2:]-self.par[1:-1])/(self.Dx[1:-1])+0.5*(self.par[1:-1]-self.par[0:-2])/(self.Dx[1:-1])
        self.Dpar[0]=(self.par[1]-self.par[0])/self.Dx[0]
        self.Dpar[-1]=(self.par[-1]-self.par[-2])/self.Dx[-1]
        
    def plot_function_values(self,figNo,color=None):
        plot_points(figNo,411,self.X,np.zeros_like(self.F),'xk','noktalar')
        plot_points(figNo,412,self.X,self.F-np.ones_like(self.F),color,'funksiyon')
        plot_points(figNo,413,self.X,self.par,color,'parametre')
        plot_points(figNo,414,self.X,self.q,color,'ısı üretimi')
        
class GEOMETRY_2D:
    def __init__(self,Min,Max,par,q0=0,n=[10,10]): 
        self.min=Min; # başlangıç değeri, 
        self.max=Max; # son değer, 
        self.n=n; # nokta sayısı 
        tmpx=np.linspace(self.min[0],self.max[0],self.n[0])
        tmpy=np.linspace(self.min[1],self.max[1],self.n[1])
          
        self.X2D,self.Y2D=np.meshgrid(tmpx,tmpy)
        self.F2D=np.zeros_like(self.X2D)
        self.par2D=np.ones_like(self.X2D)*par
        self.q2D=np.ones_like(self.X2D)*q0
        
        self.X=np.reshape(self.X2D,[-1,1])
        self.Y=np.reshape(self.Y2D,[-1,1])
        
        self.IDx=np.linspace(0,self.n[0]-1,self.n[0],dtype=np.int32)
        self.IDy=np.linspace(0,self.n[1]-1,self.n[1],dtype=np.int32)
        
        say=0; 
        self.ID1D=np.zeros([len(self.X[:,0]),2])
        self.ID2D=np.zeros_like(self.X2D)
        for j in self.IDy:
            for i in self.IDx:
                self.ID1D[say,0]=self.IDx[i]
                self.ID1D[say,1]=self.IDy[j]
                self.ID2D[j,i]=say
                say+=1;
        # self.F=np.zeros_like(self.X)
        # self.par=np.ones_like(self.X)*par
        # self.q=np.ones_like(self.X)*q0
        
    def set_function_value(self,new_value,condition,Type='f'):#lambda x: x==0    # f: func value p: parameter value 
        # koşul yazılırken değişken x olarak girilmeli 
        for j in self.IDy:
            for i in self.IDx:
                x=self.X2D[j,i]
                y=self.Y2D[j,i]
                if condition(x,y):
                    print('nokta bulundu x:%d  y:%d',x,y)
                    if Type=='f':
                        self.F2D[j,i]=(new_value)
                    elif Type=='q':
                        self.q2D[j,i]=new_value
                    elif Type=='p':
                        self.par2D[j,i]=new_value
                    else: 
                        print('Hatalı işlem yaptınız')
                        
    def flat_to_1D(self):
        self.F=np.reshape(self.F2D,[-1,1])
        self.par=np.reshape(self.par2D,[-1,1])
        self.Dpar_x=np.reshape(self.Dpar2D_x,[-1,1])
        self.Dpar_y=np.reshape(self.Dpar2D_y,[-1,1])
        self.q=np.reshape(self.q2D,[-1,1])
        
    def get_value(self,ids): 
        
        return [self.X[ids],self.Y[ids]],self.F[ids],self.par[ids],[self.Dpar_x[ids],self.Dpar_x[ids]],self.q[ids]
    
    def get_derivative_of_par(self):
        self.derivative_of_parameters=np.zeros_like(self.X)
        self.Dx2D=np.zeros_like(self.X2D)
        self.Dx2D[:,1:-1]=self.X2D[:,2:]-self.X2D[:,1:-1]
        self.Dx2D[:,0]=self.X2D[:,1]-self.X2D[:,0]
        self.Dx2D[:,-1]=self.X2D[:,-1]-self.X2D[:,-2]
        
        self.Dy2D=np.zeros_like(self.Y2D)
        self.Dy2D[1:-1,:]=self.Y2D[2:]-self.Y2D[1:-1,:]
        self.Dy2D[0,:]=self.Y2D[1]-self.Y2D[0,:]
        self.Dy2D[-1,:]=self.Y2D[-1,:]-self.Y2D[-2,:]
        
        self.Dpar2D_x=np.zeros_like(self.X2D)
        self.Dpar2D_x[:,1:-1]=0.5*(self.par2D[:,2:]-self.par2D[:,1:-1])/(self.Dx2D[:,1:-1])+0.5*(self.par2D[:,1:-1]-self.par2D[:,0:-2])/(self.Dx2D[:,1:-1])
        self.Dpar2D_x[:,0]=(self.par2D[:,1]-self.par2D[:,0])/self.Dx2D[:,0]
        self.Dpar2D_x[:,-1]=(self.par2D[:,-1]-self.par2D[:,-2])/self.Dx2D[:,-1]
        
        self.Dpar2D_y=np.zeros_like(self.Y2D)
        self.Dpar2D_y[1:-1,:]=0.5*(self.par2D[2:,:]-self.par2D[1:-1,:])/(self.Dy2D[1:-1,:])+0.5*(self.par2D[1:-1,:]-self.par2D[0:-2,:])/(self.Dy2D[1:-1,:])
        self.Dpar2D_y[0,:]=(self.par2D[1,:]-self.par2D[0,:])/self.Dy2D[0,:]
        self.Dpar2D_y[-1,:]=(self.par2D[-1,:]-self.par2D[-2,:])/self.Dy2D[-1,:]
        self.flat_to_1D()
        
    def plot_function_values_2D(self,figNo,color=None):
        plot_points(figNo,221,self.X,self.Y,'xk','noktalar')
        plot_surf_2D(figNo,222,self.X2D,self.Y2D,self.F2D-np.ones_like(self.F2D),'funksiyon')
        plot_surf_2D(figNo,223,self.X2D,self.Y2D,self.par2D,'parametre')
        plot_surf_2D(figNo,224,self.X2D,self.Y2D,self.q2D,'ısı üretimi')
    
    def plot_function_values_3D(self,figNo,color=None):
        plot_points(figNo,221,self.X,self.Y,'xk','noktalar')
        plot_surf_3D(figNo,222,self.X2D,self.Y2D,self.F2D-np.ones_like(self.F2D),'funksiyon')
        plot_surf_3D(figNo,223,self.X2D,self.Y2D,self.par2D,'parametre')
        plot_surf_3D(figNo,224,self.X2D,self.Y2D,self.q2D,'ısı üretimi')
    
def loss_Func( y_ref, y, err_type):
    diff= y-y_ref
    if err_type=='e':
        ress=diff
    elif err_type=='ae':
        ress=tf.abs(diff)
    elif err_type=='me':
        dist_sq = tf.square(diff)
        ress= tf.reduce_mean(dist_sq)
    elif err_type=='mae':
        diff_abs=tf.abs(diff)
        ress=tf.reduce_mean(diff_abs)
    elif err_type=='se':
        dist_sq = tf.square(diff)
        ress=tf.reduce_sum(dist_sq)
    elif err_type=='sae':
        dist_sq = tf.square(diff)
        ress=tf.reduce_sum(tf.abs(dist_sq))
    elif err_type=='mse':
        dist_sq = tf.square(diff)
        ress=tf.reduce_mean(dist_sq)
    elif err_type=='sse':
        dist_sq = tf.square(diff)
        ress=tf.reduce_sum(dist_sq)    
    elif err_type=='rmse':
        dist_sq = tf.square(diff)
        mean_squared_diff= tf.reduce_mean(dist_sq)
        ress=tf.sqrt(mean_squared_diff)
    
    return ress
       