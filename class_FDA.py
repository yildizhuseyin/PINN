# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:48:31 2025

@author: HY
"""

from class_Geometry import *

    
class FDA_1D:
    # Sonlu elemanlar gauss-sider yöntemi kullanarak iteratif hesaplama 
    def __init__(self,Geo): 
        self.geo=Geo
    def apply_FDA(self,epoch,err_lim=1e-3):
        geo=self.geo
        self.t0=t.time()
        for i in range(epoch): 
            tmpF=geo.F.copy()
            for j in range(1,len(geo.X)-1):
                B=geo.Dx[j]**2/(2*geo.par[j])
                A1=geo.par[j]/(geo.Dx[j]**2)+geo.Dpar[j]/(geo.Dx[j])
                A2=geo.par[j]/(geo.Dx[j]**2)-geo.Dpar[j]/(geo.Dx[j])
                geo.F[j]=B*(geo.q[j]+A1*geo.F[j+1]+A2*geo.F[j-1])
            errs=np.abs(tmpF-geo.F)
            mean_err=np.mean(errs)
        self.SIM_time=t.time()-self.t0
        print('clasical FDA simulation time : ',self.SIM_time)
        self.resoult_FEA=geo
        
    def apply_FDA_with_gauss_sider(self,epoch,lamda=1.0,err_lim=1e-3):
        geo=self.geo
        self.t0=t.time();
        for i in range(epoch): 
            tmpF=geo.F.copy()
            newF=geo.F.copy()
            for j in range(1,len(geo.X)-1):
                B=geo.Dx[j]**2/(2*geo.par[j])
                A1=geo.par[j]/(geo.Dx[j]**2)+geo.Dpar[j]/(geo.Dx[j])
                A2=geo.par[j]/(geo.Dx[j]**2)-geo.Dpar[j]/(geo.Dx[j])
                newF[j]=B*(geo.q[j]+A1*geo.F[j+1]+A2*geo.F[j-1])
                geo.F[j]=(lamda*newF[j]+(1-lamda)*tmpF[j])
            errs=np.abs(tmpF-geo.F)
            mean_err=np.mean(errs)
        self.SIM_time=t.time()-self.t0
        print('gauss-sider FDA simulation time : ',self.SIM_time)
        self.resoult_gsFDA=geo
        

    
