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
T0=20.0/Tmax # C x=0 boyutsuz sıcaklık     
TL=100.0/Tmax # C x=L boyutsuz sıcaklık    
k=100/kmax  # boyutsuz ısı iletim katsayısı 
q=0*q_n;  # boyutsuz ısı üretimi 
h=0*h_n;  # boyutsuz ısı taşınım katsayısı 

# Geometri tanımlama işlemleri  
geo=GEOMETRY_1D(0,1,k,n=15)
# Geometriyi tanımla 
geo.set_function_value(T0, lambda x: x==0,Type='f')
geo.set_function_value(TL, lambda x: x==L,Type='f')
geo.get_derivative_of_par()
# geo.plot_function_values(1,color='-b')


# Sonlu Farklar 
def heat_eq_1D(geo,j):
    B=geo.Dx[j]**2/(2*geo.par[j])
    A1=geo.par[j]/(geo.Dx[j]**2)+geo.Dpar[j]/(geo.Dx[j])
    A2=geo.par[j]/(geo.Dx[j]**2)-geo.Dpar[j]/(geo.Dx[j])
    new_value=B*(geo.q[j]+A1*geo.F[j+1]+A2*geo.F[j-1])
    return new_value
FDA=FDA_1D(geo,heat_eq_1D) # klasik iterasyonlu sonlu farklar analizi 
FDAgs=FDA_1D(geo,heat_eq_1D) # gauss - sider iterasyonu sonlu farklar analizi 

# FDA.apply_FEA(10)
# FDA.resoult_FDA.plot_function_values(2,color='-b')

FDAgs.apply_FDA_with_gauss_sider(750)
# FDAgs.resoult_gsFDA.plot_function_values(3,color='-g')




# Buradan sonra Fizik bilgili sinir ağını kuruyoruz ve eğitiyoruz. 
PINN=PINN_1D() 
# https://keras.io/api/layers/initializers/
# random_normal   random_uniform    zeros    RandomUniform(minval=0.0, maxval=1.0)
inis=keras.initializers.RandomUniform(minval=0.00, maxval=0.4, seed=None)

PINN.model.add(layers.Dense(1, activation='relu',kernel_initializer=inis, input_shape=(1,)))

PINN.model.add(layers.Dense(10, activation='tanh',kernel_initializer=inis))
PINN.model.add(layers.Dense(20, activation='tanh',kernel_initializer=inis))
PINN.model.add(layers.Dense(10, activation='tanh',kernel_initializer=inis))

# PINN.model.add(layers.Dense(5, activation='tanh',kernel_initializer=inis))

PINN.model.add(layers.Dense(1, activation='relu',kernel_initializer=inis))

PINN.model.summary() # Network özeti 
len(PINN.model.trainable_weights) # ağırlık dizisi 
PINN.model.get_weights() # ağırlıkları getir  
PINN.model.trainable_weights[0] 
len(PINN.model.trainable_variables)




def fcn_boundry(geo,U): 
    values=U[0]# fonksiyonun değeri/ the value of the function
    return values

def fcn_PDE(geo,U): 
    PDE_loss=geo.dk*U[0]+geo.k*U[2] # Diferansiyel denklem / Differential equation 
    return PDE_loss


#PINN.add_boundry(geo,[0,-1],fcn_boundry) # geometri yapısı ve indisleri sınır koşulu olarak ata 
PINN.add_boundry(geo,[0,],fcn_boundry) # geometri yapısı ve indisleri sınır koşulu olarak ata 
PINN.add_boundry(geo,[-1],fcn_boundry) # geometri yapısı ve indisleri sınır koşulu olarak ata 

print('boundry x',PINN.boundry_points.np_x)

#PINN.add_body(geo,range(1,int(geo.n/2)),fcn_PDE) # geometri yapısı ve indisleri geçiş bölgesi olarak ata 
PINN.add_body(geo,range(1,int(geo.n/2)),fcn_PDE) # geometri yapısı ve indisleri geçiş bölgesi olarak ata 
PINN.add_body(geo,range(int(geo.n/2),geo.n),fcn_PDE) # geometri yapısı ve indisleri geçiş bölgesi olarak ata 

print('body x',PINN.body_points.np_x)

XX=FDAgs.resoult_gsFDA.X # Kontrol için x değerleri 
T_FDA=FDAgs.resoult_gsFDA.F # Sonlu farklar çözümü 
PINN.add_poly(4) # 4. mertebe bir polinom çözüm tanımla  
PINN.get_poly() # Polinom katsayılarını hesapla 
PINN.Poly.add_externap_par(0.25)
Tpoly=PINN.Poly.np_predict(XX) # Polinom için sıcaklıkları hesapla 




T_0=PINN.predict(XX) # Eğitim öncesi sıcaklıklar değerlerini getir 
PINN.train(750,lr=0.001,c=[0.9,0.1],num=50,usePoly=False,errType='rmse') # 100 iterasyon koşur 
T_1=PINN.predict(XX) # Eğitim sonrası sıcaklık değerlerini getir 
w=PINN.model.get_weights()
print(w) # ağırlıkları getir  


print('body x',PINN.body_points.X)


# plot_list_points(1,111,[(XX,T_0,'-b'),
#                         (XX,T_1,'-r')])

plot_points(4,411,XX*Lmax,T_FDA*Tmax,'-b')
# plot_points(4,412,XX*Lmax,Tpoly*Tmax,'-r')
plot_points(4,413,XX*Lmax,T_0*Tmax,'-b')
plot_points(4,414,XX*Lmax,T_1*Tmax,'-r')
"""
"""

