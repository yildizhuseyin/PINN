
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras

import matplotlib.pyplot as plt
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
       