# -*- coding: utf-8 -*-
"""
Hava nüveli manyetik alan çözümü 

"""

from class_FDA import *
from class_Collocation import *


# Boyutsuzlaştırma için gerekli parametreler 
Lmax=1; # karakteristik uzunluk 
I=1; # karakteristik akım 
N=1000 ; # Sargı sayısı 
Am=0.2*0.2 # Kesit alanı 
Jmax=N*I/(Am)
Mu0=4*np.pi*1e-7# Boşluğun manyetik geçirgenliği 
Amax=Mu0*Jmax/(Lmax**2);#Mu0*Jmax/(Lmax*Lmax) # en yüksek manyetik akım yoğunluğu 
Mumax=1*Mu0
# parametreler 

L=1.0/Lmax # boyutsuz uzunluk 
Ax0=0.0/Amax # C x=0 boyutsuz sıcaklık     
Ax1=0.0/Amax # C x=L boyutsuz sıcaklık    
Ay0=0.0/Amax # C x=0 boyutsuz sıcaklık     
Ay1=0.0/Amax # C x=L boyutsuz sıcaklık    

M0=Mu0#/Mumax  # boyutsuz manyetik geçirgenlik 
M1=1000*M0 # 14872*M0
f0=1/M0
f1=1/M1
J=N*I/(Am)#/Jmax; # Boyutsuz akı

center_points=15
FDA_points=3*(center_points-1)+1

# Geometri tanımlama işlemleri  
geo=GEOMETRY_2D([0,0],[L,L],f0,n=[FDA_points,FDA_points])
# Geometriyi tanımla 
geo.set_function_value(Ax0, lambda x,y: x==0,Type='f')
geo.set_function_value(Ax1, lambda x,y: x==L,Type='f')
geo.set_function_value(Ay0, lambda x,y: 0<x<L and y==0,Type='f')
geo.set_function_value(Ay1, lambda x,y: 0<x<L and y==L,Type='f')

geo.set_function_value(J, lambda x,y: 0.4<=x<=0.6 and 0.3<=y<=0.5,Type='q')
geo_copy=geo.copy()

geo.set_function_value(f1, lambda x,y: 0.1<=x<=0.30 and 0.25<=y<=0.55,Type='p')
geo.set_function_value(f1, lambda x,y: 0.7<=x<=0.90 and 0.25<=y<=0.55,Type='p')
geo.set_function_value(f1, lambda x,y: 0.1<=x<=0.90 and 0.6<=y<=0.9,Type='p')

# geo.set_function_value((f1+f0)/2, lambda x,y: x==0.1 and 0.35<=y<=0.65,Type='p')
# geo.set_function_value((f1+f0)/2, lambda x,y: x==0.30 and 0.35<=y<=0.65,Type='p')
# geo.set_function_value((f1+f0)/2, lambda x,y: 0.1<=x<=0.25 and y==0.35,Type='p')
# geo.set_function_value((f1+f0)/2, lambda x,y: 0.1<=x<=0.25 and y==0.65,Type='p')

geo.get_derivative_of_par()
geo_copy.get_derivative_of_par()

# geo.plot_function_values_2D(1,color='-b')

global pp 
pp=1
# Sonlu Farklar 
def pde_Maxwels_equation_2D(geo,i,j): #Isı transferi için iki boyutlu sıcaklık denkleminin uygulanışı 
    global pp
    C1=+(1/(geo.Dx2D[j,i]**2))*geo.par2D[j,i]
    C2=+(1/(geo.Dy2D[j,i]**2))*geo.par2D[j,i]
    C3=+1; C4=0; C5=0
    M=2*(C1+C2)
    s1,s2=geo.par2D.shape
    if i==0: 
        new_temp_value=(1.0/M)*(-C3*geo.q2D[j,i]
                                +C1*geo.F2D[j,i+1]+C1*geo.F2D[j,i+1]
                                +C2*geo.F2D[j+1,i]+C2*geo.F2D[j-1,i]
                                +C4*geo.F2D[j+1,i]-C4*geo.F2D[j+1,i]
                                +C5*geo.F2D[j,i+1]-C5*geo.F2D[j,i+1])
    elif i>2 and j>2 and i<s1-2 and j<s2-2:
        C1b=+2*0.5*(1/(geo.Dx2D[j,i+1]**2))*geo.par2D[j,i+1]
        C1a=+2*0.5*(1/(geo.Dx2D[j,i-1]**2))*geo.par2D[j,i-1]
        C2b=+2*0.5*(1/(geo.Dy2D[j+1,i]**2))*geo.par2D[j+1,i]
        C2a=+2*0.5*(1/(geo.Dy2D[j-1,i]**2))*geo.par2D[j-1,i]
        C4b=+0*0.5*(1/geo.Dx2D[j,i+1])*geo.Dpar2D_x[j,i+1]
        C4a=+0*0.5*(1/geo.Dx2D[j,i-1])*geo.Dpar2D_x[j,i-1]
        C5b=+0*0.5*(1/geo.Dy2D[j+1,i])*geo.Dpar2D_y[j+1,i]
        C5a=+0*0.5*(1/geo.Dy2D[j-1,i])*geo.Dpar2D_y[j-1,i]
        C3=+1.0
        M=1*(C1b+C1a+C2b+C2a+C4b+C4a+C5b+C5a)
        # 2	−5	4	−1
        new_temp_value=(1.0/M)*(+C3*geo.q2D[j,i]
                                +C1b*(1*geo.F2D[j,i+1])
                                +C1a*(1*geo.F2D[j,i-1])
                                +C2b*(1*geo.F2D[j+1,i])
                                +C2a*(1*geo.F2D[j-1,i])
                                +C4b*(1*geo.F2D[j,i+1])
                                +C4a*(1*geo.F2D[j,i-1])
                                +C5b*(1*geo.F2D[j+1,i])
                                +C5a*(1*geo.F2D[j-1,i]))

    else: 
        new_temp_value=(1.0/M)*(+C3*geo.q2D[j,i]
                                +C1*geo.F2D[j,i+1]+C1*geo.F2D[j,i-1]
                                +C2*geo.F2D[j+1,i]+C2*geo.F2D[j-1,i]
                                +C4*geo.F2D[j+1,i]-C4*geo.F2D[j-1,i]
                                +C5*geo.F2D[j,i+1]-C5*geo.F2D[j,i-1])
    #print(C5,C5,M,new_temp_value,type(new_temp_value))
    return new_temp_value
    
"""

FDA=FDA_2D(geo_copy,pde_Maxwels_equation_2D) # klasik iterasyonlu sonlu farklar analizi 
FDA.apply_FDA(500)
# FDA.resoult_FDA.plot_function_values_2D(1,color='-b')
# F2D=FDA.resoult_FDA.F2D
B=FDA.apply_cross_product()
# FDA.resoult_FDA.plot_stream_line(5,131)
# FDA.resoult_FDA.plot_magnetic_fields(3)
# FDA.plot_log(6,211)
"""

FDAgs=FDA_2D(geo,pde_Maxwels_equation_2D) # gauss - sider iterasyonu sonlu farklar analizi 
# FDAgs.set_fonction_values(FDA.geo.F2D)
# FDAgs.set_more_calculation_area('x0') # x=0 da simetri koşulu dT/dx=0 

# FDAgs.apply_FDA(500)
FDAgs.apply_FDA_with_gauss_sider(100,lamda=0.99)
FDAgs.resoult_FDA.plot_function_values_2D(2,color='-g')
# F2Dgs=FDAgs.resoult_FDA.F2D
Bgs=FDAgs.apply_cross_product()


XX=FDAgs.resoult_FDA.X2D # Kontrol için x değerleri 
YY=FDAgs.resoult_FDA.Y2D # Kontrol için x değerleri 
T_FDA=FDAgs.resoult_FDA.F2D # Sonlu farklar çözümü 

# FDAgs.resoult_FDA.plot_stream_line(5,132)
FDAgs.resoult_FDA.plot_magnetic_fields(4)
# FDAgs.plot_log(6,212)

FDA=FDAgs


## Buradan sonra Polinom Kollokasyonu Uygulanacak 
Coll_RBF2D=collocation_2D(par=['mquad'],Type='rbf')
Coll_RBF2D.FCN.set_function_type('gauss') # gauss mquad  iquad imquad eUzal

# Kollokasyon noktalarını tanımla
geo_center_points=GEOMETRY_2D([0,0],[L,L],f0,n=[center_points,center_points])
# geo_center_points=GEOMETRY_2D([0,0],[L,L],f0,n=[5,5])

geo_center_points.get_derivative_of_par()

# Coll_RBF2D.RBF_add_center_point_with_points(0.5, 0.5, 6, 6)# xm, ym, ex, ey
cond_center_points=lambda x,y: x>0 and x<L and y>0 and y<L
Coll_RBF2D.RBF_add_center_point_with_geo(geo_center_points, cond_center_points,1.0)#geo, condition, par

Coll_RBF2D.FCN.Xm
Coll_RBF2D.FCN.Ex


def fcn_boundry(geo,model):  # x=L, y=0, y=L ==> T=0 
    Matrix=model.get_matrix(geo.X,geo.Y,der='f')
    Resoults=tf.zeros_like(geo.F)
    return Matrix,Resoults

def fcn_boundry_fx(geo,model):  # x=L, y=0, y=L ==> T=0 
    Matrix=model.get_matrix(geo.X,geo.Y,der='fx')
    Resoults=tf.zeros_like(geo.F)
    return Matrix,Resoults

def fcn_boundry_fy(geo,model):  # x=L, y=0, y=L ==> T=0 
    Matrix=model.get_matrix(geo.X,geo.Y,der='fy')
    Resoults=tf.zeros_like(geo.F)
    return Matrix,Resoults

def fcn_PDE(geo,model): # U=[[U],[dU/dx,dU/dy], [d2U/dx2,d2U/dxdy,d2U/dy2]]
    Uxx=model.get_matrix(geo.X,geo.Y,der='fxx')
    Uyy=model.get_matrix(geo.X,geo.Y,der='fyy')
    Ux=model.get_matrix(geo.X,geo.Y,der='fx')
    Uy=model.get_matrix(geo.X,geo.Y,der='fy')
    Matrix=-(geo.k*model.Ir)*(Uxx+Uyy) # Diferansiyel denklem / Differential equation 
    #Matrix=-(geo.dk_dx*model.Ir)*Ux-(geo.dk_dy*model.Ir)*Uy-(geo.k*model.Ir)*(Uxx+Uyy) # Diferansiyel denklem / Differential equation 
    Resoults=geo.q
    return Matrix,Resoults

def fcn_jump(geo,model):  # x=L, y=0, y=L ==> T=0 
# curl(A) U_y i -U_x
    Ux=model.get_matrix(geo.X,geo.Y,der='fx')
    Uy=model.get_matrix(geo.X,geo.Y,der='fy')
    Dk=(geo.k-geo.k2)*model.Ir
    ax=(geo.nx*model.Ir)
    ay=(geo.ny*model.Ir)
    #Matrix=Dk*(ax*Ux+ay*Uy)
    Matrix=Dk*(Uy*ax-Ux*ay)
    Resoults=tf.zeros_like(geo.F)
    return Matrix,Resoults


# Sınır koşullarını belirle 
cond_boundry_x0=lambda x,y: x==0
cond_boundry_xL=lambda x,y: x==L
cond_boundry_y0=lambda x,y: y==0
cond_boundry_yL=lambda x,y: y==L

Coll_RBF2D.add_boundry(FDA.resoult_FDA,fcn_boundry,cond_boundry_x0,color='xb') # x=0 için bütün noktalar 
Coll_RBF2D.add_boundry(FDA.resoult_FDA,fcn_boundry,cond_boundry_xL) # x=L için bütün noktalar 
Coll_RBF2D.add_boundry(FDA.resoult_FDA,fcn_boundry,cond_boundry_y0) # y=0 için bütün noktalar 
Coll_RBF2D.add_boundry(FDA.resoult_FDA,fcn_boundry,cond_boundry_yL) # y=L için bütün noktalar 
print('boundry count',Coll_RBF2D.number_of_boundry_geometry)

# Gövdeleri belirle 
# cond_body=lambda x,y: x>0 and x<L and y>0 and y<L

cond_body0=lambda x,y: ((0<x<L and 0<y<L) and not (0.1<=x<=0.3 and 0.25<=y<=0.55) and not (0.7<=x<=0.9 and 0.25<=y<=0.55) and not (0.1<=x<=0.9 and 0.6<=y<=0.9) and not (0.4<=x<=0.6 and 0.3<=y<=0.5))
cond_body1=lambda x,y: ((0<x<L and 0<y<L) and ((0.12<x<0.28 and 0.27<y<0.53) or (0.72<x<0.88 and 0.28<y<0.53) or (0.12<x<0.88 and 0.62<y<0.88) ))
cond_coil=lambda x,y: ((0<x<L and 0<y<L) and (0.4<=x<=0.6 and 0.3<=y<=0.5))

Coll_RBF2D.add_body(geo,fcn_PDE,cond_body0,color='.b') # y=L için bütün noktalar
Coll_RBF2D.add_body(geo,fcn_PDE,cond_body1,color='.k') # y=L için bütün noktalar 
Coll_RBF2D.add_body(geo,fcn_PDE,cond_coil,color='.y') # y=L için bütün noktalar 
# Coll_RBF2D.add_body(geo,fcn_PDE,cond_body3) # y=L için bütün noktalar 
# Coll_RBF2D.add_body(geo,fcn_PDE,cond_body4) # y=L için bütün noktalar 
print('body count',Coll_RBF2D.number_of_body_geometry)


cond_jump1_x=lambda x,y: ((0.09<x<0.11 and 0.26<y<0.53) or (0.29<x<0.31 and 0.26<y<0.53) )
cond_jump1_y=lambda x,y: ((0.11<x<0.28 and 0.23<y<0.26) or (0.11<x<0.28 and 0.53<y<0.57) )
cond_jump2_x=lambda x,y: ((0.68<x<0.72 and 0.26<y<0.53) or (0.89<x<0.91 and 0.26<y<0.53) )
cond_jump2_y=lambda x,y: ((0.72<x<0.86 and 0.23<y<0.26) or (0.72<x<0.88 and 0.53<y<0.57) )
cond_jump3_x=lambda x,y: ((0.08<x<0.12 and 0.72<y<0.87) or (0.88<x<0.91 and 0.62<y<0.87) )
cond_jump3_y=lambda x,y: ((0.12<x<0.86 and 0.68<y<0.72) or (0.12<x<0.88 and 0.88<y<0.92) )



Coll_RBF2D.add_jump_geometry(geo,fcn_jump,cond_jump1_x,[f0,f1],[-1,0],color='sk')
Coll_RBF2D.add_jump_geometry(geo,fcn_jump,cond_jump1_y,[f0,f1],[0,-1],color='sk')
Coll_RBF2D.add_jump_geometry(geo,fcn_jump,cond_jump2_x,[f0,f1],[-1,0],color='sk')
Coll_RBF2D.add_jump_geometry(geo,fcn_jump,cond_jump2_y,[f0,f1],[0,-1],color='sk')
Coll_RBF2D.add_jump_geometry(geo,fcn_jump,cond_jump3_x,[f0,f1],[-1,0],color='sk')
Coll_RBF2D.add_jump_geometry(geo,fcn_jump,cond_jump3_y,[f0,f1],[0,-1],color='sk')

#Coll_RBF2D.add_jump_geometry(geo,fcn_jump,cond_jump_x0,[f0,f1],[1,0],color='sb')

print('jump count',Coll_RBF2D.list_of_jump_geometry)

# plot_points(1,111,Coll_RBF2D.list_of_body_geometry[0].np_x,Coll_RBF2D.list_of_body_geometry[0].np_y,color='.k')
# plot_points(1,111,Coll_RBF2D.list_of_body_geometry[1].np_x,Coll_RBF2D.list_of_body_geometry[1].np_y,color='.b')
# plot_list_points(1,111,[(Coll_RBF2D.list_of_body_geometry[0].np_x,Coll_RBF2D.list_of_body_geometry[0].np_y,'.k'),
#                         (Coll_RBF2D.list_of_body_geometry[1].np_x,Coll_RBF2D.list_of_body_geometry[1].np_y,'.b'),
#                         (Coll_RBF2D.list_of_jump_geometry[0].np_x,Coll_RBF2D.list_of_jump_geometry[0].np_y,'sb'),
#                         (Coll_RBF2D.list_of_jump_geometry[1].np_x,Coll_RBF2D.list_of_jump_geometry[1].np_y,'sb'),
#                         (Coll_RBF2D.list_of_jump_geometry[2].np_x,Coll_RBF2D.list_of_jump_geometry[2].np_y,'sg'),
#                         (Coll_RBF2D.list_of_jump_geometry[3].np_x,Coll_RBF2D.list_of_jump_geometry[3].np_y,'sg'),
#                         (Coll_RBF2D.list_of_jump_geometry[4].np_x,Coll_RBF2D.list_of_jump_geometry[4].np_y,'sy'),
#                         (Coll_RBF2D.list_of_jump_geometry[5].np_x,Coll_RBF2D.list_of_jump_geometry[5].np_y,'sy')])

pp=Coll_RBF2D.list_of_body_geometry[0]

# Kollokasyon uygulama 
Coll_RBF2D.apply_collocation()
Coll_RBF2D.FCN.C

XX=FDA.resoult_FDA.X2D[1:-1,1:-1]
YY=FDA.resoult_FDA.Y2D[1:-1,1:-1]

FDA.resoult_FDA.plot_function_2D(7,231)
FDA.resoult_FDA.plot_stream_line(7,234)

Coll_RBF2D.plot_function(7, 232, XX, YY)
Coll_RBF2D.plot_stream_line(7, 235, XX, YY)
Coll_RBF2D.plot_curl_2D(8, XX, YY)

Coll_RBF2D.FCN.C=tf.abs(Coll_RBF2D.FCN.C)
  


# Model non-lineer parameter optimization 
C_first=Coll_RBF2D.FCN.C
# Coll_RBF2D.FCN.set_tranible_parameters(Xm=True,Ym=True,Ex=False,Ey=False,C=False)
# Coll_RBF2D.train(250,lr=0.005,c=[0.1,0.9],num=10,errType='rmse') # 100 iterasyon koşur 
# Coll_RBF2D.apply_collocation()

oran=[0.1,0.4,0.75]
Coll_RBF2D.FCN.set_tranible_parameters(Xm=False,Ym=False,Ex=True,Ey=True,C=False)
Coll_RBF2D.train(500,lr=0.1,c=oran,num=10,errType='rmse',reset=True,title='shape parameters') # 100 iterasyon koşur 

Coll_RBF2D.train(200,lr=0.05,c=oran,num=10,errType='rmse',reset=True,title='shape parameters') # 100 iterasyon koşur 



for i in range(10):
    # Coll_RBF2D.error_analysis_on_geometry(geo,fcn_PDE,9)

    Coll_RBF2D.FCN.set_tranible_parameters(Xm=False,Ym=False,Ex=True,Ey=True,C=False)
    Coll_RBF2D.train(100,lr=0.01,c=oran,num=10,errType='rse',reset=False,title='shape parameters') # 100 iterasyon koşur 
    
    # Coll_RBF2D.FCN.set_tranible_parameters(Xm=True,Ym=True,Ex=False,Ey=False,C=False)
    # Coll_RBF2D.train(100,lr=1e-2,c=oran,num=10,errType='mse',reset=False,title='center points') # 100 iterasyon koşur 
    # # # Coll_RBF2D.apply_collocation()
    
    Coll_RBF2D.FCN.set_tranible_parameters(Xm=False,Ym=False,Ex=False,Ey=False,C=True)
    Coll_RBF2D.train(100,lr=1e-7,c=oran,num=10,errType='rse',reset=False,title='weights') # 100 iterasyon koşur 
    # Coll_RBF2D.apply_collocation()
    


Tvars=Coll_RBF2D.FCN.trainable_variables

# Coll_RBF2D.train(500,lr=0.001,c=[0.5,0.5],num=10,errType='mse') # 100 iterasyon koşur 
C_last=Coll_RBF2D.FCN.C
C_=np.concatenate((C_first, C_last),axis=1)
E=np.concatenate((Coll_RBF2D.FCN.Ex.numpy(),Coll_RBF2D.FCN.Ey.numpy()),axis=0)

Coll_RBF2D.plot_function(7, 233, XX, YY)
Coll_RBF2D.plot_stream_line(7, 236, XX, YY)
Coll_RBF2D.plot_curl_2D(9, XX, YY)
Coll_RBF2D.plot_log(10)

CC=Coll_RBF2D.FCN.C.numpy()
CCex=Coll_RBF2D.FCN.Ex.numpy()
CCey=Coll_RBF2D.FCN.Ey.numpy()

"""
"""

"""
"""
