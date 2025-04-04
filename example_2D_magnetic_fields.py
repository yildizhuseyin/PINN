# -*- coding: utf-8 -*-
"""
Hava nüveli manyetik alan çözümü 

"""

from class_FDA import *
from class_PINN import *

# Boyutsuzlaştırma için gerekli parametreler 
Lmax=1; # karakteristik uzunluk 
I=10; # karakteristik akım 
N=1000 ; # Sargı sayısı 
Am=0.25*0.25 # Kesit alanı 
Jmax=N*I/(Am)
Mu0=4*np.pi*1e-7# Boşluğun manyetik geçirgenliği 
Amax=Mu0*Jmax/(Lmax**2);#Mu0*Jmax/(Lmax*Lmax) # en yüksek manyetik akım yoğunluğu 

# parametreler 

L=1.0/Lmax # boyutsuz uzunluk 
Ax0=0.0/Amax # C x=0 boyutsuz sıcaklık     
Ax1=0.0/Amax # C x=L boyutsuz sıcaklık    
Ay0=0.0/Amax # C x=0 boyutsuz sıcaklık     
Ay1=0.0/Amax # C x=L boyutsuz sıcaklık    

M1=Mu0/Mu0  # boyutsuz manyetik geçirgenlik 
f=1/M1
J=N*I/(Am)/Jmax; # Boyutsuz akı

# Geometri tanımlama işlemleri  
geo=GEOMETRY_2D([0,0],[L,L],f,n=[15,15])
# Geometriyi tanımla 
geo.set_function_value(Ax0, lambda x,y: x==0,Type='f')
geo.set_function_value(Ax1, lambda x,y: x==L,Type='f')
geo.set_function_value(Ay0, lambda x,y: 0<x<L and y==0,Type='f')
geo.set_function_value(Ay1, lambda x,y: 0<x<L and y==L,Type='f')

geo.set_function_value(J, lambda x,y: 0.35<=x<=0.65 and 0.35<=y<=0.65,Type='q')

geo.get_derivative_of_par()
geo.plot_function_values_2D(1,color='-b')

# Sonlu Farklar 
def pde_heat_2D(geo,i,j): #Isı transferi için iki boyutlu sıcaklık denkleminin uygulanışı 
    C1=(geo.Dy2D[j,i]**2)*geo.par2D[j,i]
    C2=(geo.Dx2D[j,i]**2)*geo.par2D[j,i]
    C3=(geo.Dx2D[j,i]**2+geo.Dy2D[j,i]**2)
    #C1=(geo.Dx2D[j,i])*(geo.Dy2D[j,i]**2)*geo.Dpar2D_x[j,i]-(geo.Dy2D[j,i]**2)*geo.par2D[j,i]

    M=2*geo.par2D[j,i]*(geo.Dx2D[j,i]**2+geo.Dy2D[j,i]**2)
    if i==0: 
        new_temp_value=(1.0/M)*(C3*geo.q2D[j,i]
                                +C1*geo.F2D[j,i+1]+C1*geo.F2D[j,i+1]
                                +C2*geo.F2D[j+1,i]+C2*geo.F2D[j-1,i])
    else: 
        new_temp_value=(1.0/M)*(C3*geo.q2D[j,i]
                                +C1*geo.F2D[j,i+1]+C1*geo.F2D[j,i-1]
                                +C2*geo.F2D[j+1,i]+C2*geo.F2D[j-1,i])
    return new_temp_value
    

# Aşağıda denklemlerde bir hata var gibi duruyor tekrar irdelenecek. 
def pde_heat_2Da(geo,i,j): #Maxwell için iki boyutlu sıcaklık denkleminin uygulanışı 
    C1=(geo.Dx2D[j,i])*(geo.Dy2D[j,i]**2)*geo.Dpar2D_x[j,i]-(geo.Dy2D[j,i]**2)*geo.par2D[j,i]
    C2=-(geo.Dx2D[j,i]**2)*(geo.Dy2D[j,i])*geo.Dpar2D_y[j,i]-(geo.Dx2D[j,i]**2)*geo.par2D[j,i]
    C3=-(geo.Dx2D[j,i])*(geo.Dy2D[j,i]**2)*geo.Dpar2D_x[j,i]-(geo.Dy2D[j,i]**2)*geo.par2D[j,i]
    C4=+(geo.Dx2D[j,i]**2)*(geo.Dy2D[j,i])*geo.Dpar2D_y[j,i]-(geo.Dx2D[j,i]**2)*geo.par2D[j,i]
    C5=2*(geo.Dx2D[j,i]**2)*(geo.Dy2D[j,i]**2)
    M=-2*geo.par2D[j,i]*(geo.Dx2D[j,i]**2+geo.Dy2D[j,i]**2)
    new_temp_value=(1.0/M)*(-C5*geo.q2D[j,i]
                            +C1*geo.F2D[j,i+1]-C3*geo.F2D[j,i-1]
                            +C2*geo.F2D[j+1,i]-C4*geo.F2D[j-1,i])
    return new_temp_value

# FDA=FDA_2D(geo,pde_heat_2D) # klasik iterasyonlu sonlu farklar analizi 
# FDA.apply_FDA(50)
# FDA.resoult_FDA.plot_function_values_2D(2,color='-b')
# F2D=FDA.resoult_FDA.F2D
# B=FDA.apply_cross_product()

FDAgs=FDA_2D(geo,pde_heat_2D) # gauss - sider iterasyonu sonlu farklar analizi 
FDAgs.set_more_calculation_area('x0') # x=0 da simetri koşulu dT/dx=0 

FDAgs.apply_FDA_with_gauss_sider(5500,lamda=0.5)
FDAgs.resoult_FDA.plot_function_values_2D(3,color='-g')
F2Dgs=FDAgs.resoult_FDA.F2D
Bgs=FDAgs.apply_cross_product()


XX=FDAgs.resoult_FDA.X2D # Kontrol için x değerleri 
YY=FDAgs.resoult_FDA.Y2D # Kontrol için x değerleri 
T_FDA=FDAgs.resoult_FDA.F2D # Sonlu farklar çözümü 

FDAgs.resoult_FDA.plot_stream_line(4,111)
FDAgs.resoult_FDA.plot_magnetic_fields(5)


# Buradan sonra Fizik bilgili sinir ağını kuruyoruz ve eğitiyoruz. 
PINN=PINN_2D() 
# https://keras.io/api/layers/initializers/
# random_normal   random_uniform    zeros    RandomUniform(minval=0.0, maxval=1.0)
inis=keras.initializers.RandomUniform(minval=0.00, maxval=0.1, seed=None)
# inis=keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None)

PINN.model.add(layers.Dense(5, activation='tanh',kernel_initializer=inis, input_shape=(2,)))

PINN.model.add(layers.Dense(5, activation='sigmoid',kernel_initializer=inis))
# PINN.model.add(layers.Dense(10, activation='tanh',kernel_initializer=inis))
# PINN.model.add(layers.Dense(25, activation='tanh',kernel_initializer=inis))

PINN.model.add(layers.Dense(5, activation='tanh',kernel_initializer=inis))

# PINN.model.add(layers.Dense(5, activation='tanh',kernel_initializer=inis))

PINN.model.add(layers.Dense(1, activation='tanh',kernel_initializer=inis))

PINN.model.summary() # Network özeti 
len(PINN.model.trainable_weights) # ağırlık dizisi 
w=PINN.model.get_weights() # ağırlıkları getir  
PINN.model.trainable_weights[0] 
len(PINN.model.trainable_variables)




def fcn_boundry(geo,U):  # x=L, y=0, y=L ==> T=0 
    # T(x,y)
    values=U[0][0]# fonksiyonun değeri/ the value of the function
    err=tf.zeros_like(geo.F)-values
    return err

def fcn_PDE(geo,U): # U=[[U],[dU/dx,dU/dy], [d2U/dx2,d2U/dxdy,d2U/dy2]]
    #pde=dk_dx*Ux+dk_dy*Uy+k*(Uxx+Uyy)
    #PDE_loss=(geo.dk_dx*U[1][0]+geo.dk_dy*U[1][1])+geo.k*(U[2][0]+U[2][2])
    PDE_loss=geo.q+geo.k*(U[2][0]+U[2][2]) # Diferansiyel denklem / Differential equation 
    # a=U[0][0].numpy()
    # ax=U[1][0].numpy()
    # ay=U[1][1].numpy()
    # axx=U[2][0].numpy()
    # axy=U[2][1].numpy()
    # ayy=U[2][2].numpy()
    return PDE_loss

# Sınır koşullarını belirle 
cond_boundry_x0=lambda x,y: x==0
cond_boundry_xL=lambda x,y: x==L
cond_boundry_y0=lambda x,y: y==0
cond_boundry_yL=lambda x,y: y==L

PINN.add_boundry(geo,fcn_boundry,cond_boundry_x0,color='xb') # x=0 için bütün noktalar 
PINN.add_boundry(geo,fcn_boundry,cond_boundry_xL) # x=L için bütün noktalar 
PINN.add_boundry(geo,fcn_boundry,cond_boundry_y0) # y=0 için bütün noktalar 
PINN.add_boundry(geo,fcn_boundry,cond_boundry_yL) # y=L için bütün noktalar 
print('boundry x',PINN.boundry_points.np_x)

# Gövdeleri belirle 
cond_body=lambda x,y: x>0 and x<L and y>0 and y<L
PINN.add_body(geo,fcn_PDE,cond_body) # y=L için bütün noktalar 
print('body x',PINN.body_points.np_x)

PINN.plot_points_values_2D(7,221)

T_0=PINN.predict(XX,YY) # Eğitim öncesi sıcaklıklar değerlerini getir 

PINN.train(1000,lr=0.01,c=[0.5,0.5],num=10,usePoly=False,errType='sse') # 100 iterasyon koşur 

PINN.train(1000,lr=0.01,c=[0.5,0.5],num=10,usePoly=False,errType='mse') # 100 iterasyon koşur 
PINN.train(1000,lr=0.001,c=[0.5,0.5],num=50,usePoly=False,errType='mse') # 100 iterasyon koşur 

# PINN.train(500,lr=0.001,c=[0.5,0.5],num=100,usePoly=False,errType='mse') # 100 iterasyon koşur 
# PINN.train(1500,lr=0.001,c=[0.5,0.5],num=100,usePoly=False,errType='rmse') # 100 iterasyon koşur 
# PINN.train(1500,lr=0.0001,c=[0.5,0.5],num=100,usePoly=False,errType='mse') # 100 iterasyon koşur 

T_1=PINN.predict(XX,YY) # Eğitim sonrası sıcaklık değerlerini getir 
B=PINN.apply_curl(XX,YY) # Eğitim sonrası sıcaklık değerlerini getir 
PINN.plot_stream_line(5,133,XX,YY)
w_1=PINN.model.get_weights()

plot_surf_2D(7,222,XX,YY,T_FDA,'FDA')
plot_surf_2D(7,223,XX,YY,T_0,'first PINN')
plot_surf_2D(7,224,XX,YY,T_1,'last PINN')

plot_surf_3D(8,222,XX,YY,T_FDA,'FDA')
plot_surf_3D(8,223,XX,YY,T_0,'first PINN')
plot_surf_3D(8,224,XX,YY,T_1,'last PINN')

PINN.plot_log(9)