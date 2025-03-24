# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:48:31 2025

@author: HY
"""

from class_Geometry import *

    
def example_heat_eq_1D(geo,j):
    B=geo.Dx[j]**2/(2*geo.par[j])
    A1=geo.par[j]/(geo.Dx[j]**2)+geo.Dpar[j]/(geo.Dx[j])
    A2=geo.par[j]/(geo.Dx[j]**2)-geo.Dpar[j]/(geo.Dx[j])
    new_value=B*(geo.q[j]+A1*geo.F[j+1]+A2*geo.F[j-1])
    return new_value

class FDA_1D:
    # Sonlu elemanlar gauss-sider yöntemi kullanarak iteratif hesaplama 
    def __init__(self,Geo,func): 
        self.geo=Geo
        self.fcn=func
    def apply_FDA(self,epoch,err_lim=1e-3):
        geo=self.geo
        self.t0=t.time()
        for i in range(epoch): 
            tmpF=geo.F.copy()
            for j in range(1,len(geo.X)-1):
                geo.F[j]=self.fcn(geo,j)
            errs=np.abs(tmpF-geo.F)
            mean_err=np.mean(errs)
        self.SIM_time=t.time()-self.t0
        print('clasical FDA simulation time : ',self.SIM_time)
        self.resoult_FDA=geo
        
    def apply_FDA_with_gauss_sider(self,epoch,lamda=1.0,err_lim=1e-3):
        geo=self.geo
        self.t0=t.time();
        for i in range(epoch): 
            tmpF=geo.F.copy()
            newF=geo.F.copy()
            for j in range(1,len(geo.X)-1):
                newF[j]=self.fcn(geo,j)
                geo.F[j]=(lamda*newF[j]+(1-lamda)*tmpF[j])
            errs=np.abs(tmpF-geo.F)
            mean_err=np.mean(errs)
        self.SIM_time=t.time()-self.t0
        print('gauss-sider FDA simulation time : ',self.SIM_time)
        self.resoult_gsFDA=geo
        

def example_pde_heat2D(geo,i,j): #partial_diff_equation_for_heat_transfer
    K1x=(1/(geo.Dx2D[j,i]**2))+(1.0/geo.par2D[j,i])*(geo.Dpar2D_x[j,i])/(0.5/geo.Dx2D[j,i])
    K2x=(1/(geo.Dx2D[j,i]**2))-(1.0/geo.par2D[j,i])*(geo.Dpar2D_x[j,i])/(0.5/geo.Dx2D[j,i])
    K1y=(1/(geo.Dy2D[j,i]**2))+(1.0/geo.par2D[j,i])*(geo.Dpar2D_y[j,i])/(0.5/geo.Dy2D[j,i])
    K2y=(1/(geo.Dy2D[j,i]**2))-(1.0/geo.par2D[j,i])*(geo.Dpar2D_y[j,i])/(0.5/geo.Dy2D[j,i])
    M=(2/(geo.Dx2D[j,i]**2)+2/(geo.Dy2D[j,i]**2))
    new_value=(1.0/M)*(geo.q2D[j,i]/geo.par2D[j,i]+K1x*geo.F2D[j,i+1]+K2x*geo.F2D[j,i-1]+K1y*geo.F2D[j+1,i]+K2y*geo.F2D[j-1,i])
    return new_value
    
class FDA_2D:
    # Sonlu elemanlar gauss-sider yöntemi kullanarak iteratif hesaplama 
    def __init__(self,Geo,fonc): 
        self.geo=Geo
        self.fcn=fonc
    def apply_FDA(self,epoch,err_lim=1e-3):
        geo=self.geo
        self.t0=t.time()
        for itr in range(epoch): 
            tmpF2D=geo.F2D.copy()
            for j in geo.IDy[1:-1]:
                for i in geo.IDx[1:-1]: 
                    geo.F2D[j,i]=self.fcn(geo,i,j)
                    geo.F2D[j,i]
            errs=np.abs(tmpF2D-geo.F2D)
            mean_err=np.mean(errs)
        self.SIM_time=t.time()-self.t0
        print('clasical FDA simulation time : ',self.SIM_time)
        self.resoult_FDA=geo
        
    def apply_FDA_with_gauss_sider(self,epoch,lamda=1.0,err_lim=1e-3):
        geo=self.geo
        self.t0=t.time();
        for itr in range(epoch): 
            tmpF2D=geo.F2D.copy()
            newF2D=geo.F2D.copy()
            for j in geo.IDy[1:-1]:
                for i in geo.IDx[1:-1]:                    
                    newF2D[j,i]=val=self.fcn(geo,i,j)
                    geo.F2D[j,i]=(lamda*newF2D[j,i]+(1-lamda)*tmpF2D[j,i])
            errs=np.abs(tmpF2D-geo.F2D)
            mean_err=np.mean(errs)
        self.SIM_time=t.time()-self.t0
        print('gauss-sider FDA simulation time : ',self.SIM_time)
        self.resoult_gsFDA=geo
        

    