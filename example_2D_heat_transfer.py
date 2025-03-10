# -*- coding: utf-8 -*-
"""
Sabit sıcaklıkta daimi rejim ısı geçişi probleminin çözümü 
x=0 da T=T0, x=L de T=TL ve ısı üretimi olmayan (q=0) 
"""

from class_FDA import *
from class_PINN import *

# Boyutsuzlaştırma için gerekli parametreler 
Lmax=10; # karakteristik uzunluk 
kmax=1500; # karakteristik uzunluk 
Tmax=100; # en yüksek sıcaklık 
q_n=Lmax*Lmax/(kmax*Tmax); # Isı üretimi için boyutsuzlaştırma katsayısı 
h_n=Lmax/kmax; # taşınım sınır şartı için boyutsuzlaştırma katsayısı 


# parametreler 
L=10.0/Lmax # boyutsuz uzunluk 
Tx0=20.0/Tmax # C x=0 boyutsuz sıcaklık     
Tx1=100.0/Tmax # C x=L boyutsuz sıcaklık    
Ty0=50.0/Tmax # C x=0 boyutsuz sıcaklık     
Ty1=75.0/Tmax # C x=L boyutsuz sıcaklık    

k=100/kmax  # boyutsuz ısı iletim katsayısı 
q=0*q_n;  # boyutsuz ısı üretimi 
h=0*h_n;  # boyutsuz ısı taşınım katsayısı 

# Geometri tanımlama işlemleri  
geo=GEOMETRY_2D([0,0],[L,L],k,n=[10,15])
# Geometriyi tanımla 
geo.set_function_value(Tx0, lambda x,y: x==0,Type='f')
geo.set_function_value(Tx1, lambda x,y: x==L,Type='f')
geo.set_function_value(Ty0, lambda x,y: 0<x<L and y==0,Type='f')
geo.set_function_value(Ty1, lambda x,y: 0<x<L and y==L,Type='f')
geo.get_derivative_of_par()
geo.plot_function_values_2D(1,color='-b')

# Sonlu Farklar 
FDA=FDA_2D(geo) # klasik iterasyonlu sonlu farklar analizi 
FDAgs=FDA_2D(geo) # gauss - sider iterasyonu sonlu farklar analizi 

FDA.apply_FDA(5000)
FDA.resoult_FDA.plot_function_values_2D(2,color='-b')
F2D=FDA.resoult_FDA.F2D

FDAgs.apply_FDA_with_gauss_sider(5000)
FDAgs.resoult_gsFDA.plot_function_values_2D(3,color='-g')
F2Dgs=FDAgs.resoult_gsFDA.F2D
