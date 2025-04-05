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

M0=Mu0/Mumax  # boyutsuz manyetik geçirgenlik 
M1=1000*M0 # 14872*M0
f0=1/M0
f1=1/M1
J=N*I/(Am)/Jmax; # Boyutsuz akı

# Geometri tanımlama işlemleri  
geo=GEOMETRY_2D([0,0],[L,L],f0,n=[30,30])
# Geometriyi tanımla 
geo.set_function_value(Ax0, lambda x,y: x==0,Type='f')
geo.set_function_value(Ax1, lambda x,y: x==L,Type='f')
geo.set_function_value(Ay0, lambda x,y: 0<x<L and y==0,Type='f')
geo.set_function_value(Ay1, lambda x,y: 0<x<L and y==L,Type='f')

geo.set_function_value(J, lambda x,y: 0.4<=x<=0.6 and 0.4<=y<=0.6,Type='q')
geo_copy=geo.copy()

geo.set_function_value(f1, lambda x,y: 0.1<=x<=0.30 and 0.35<=y<=0.65,Type='p')
# geo.set_function_value(f1, lambda x,y: 0.7<=x<=0.90 and 0.35<=y<=0.65,Type='p')
# geo.set_function_value(f1, lambda x,y: 0.1<=x<=0.90 and 0.7<=y<=0.8,Type='p')

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
FDA.resoult_FDA.plot_function_values_2D(1,color='-b')
# F2D=FDA.resoult_FDA.F2D
B=FDA.apply_cross_product()
FDA.resoult_FDA.plot_stream_line(5,131)
FDA.resoult_FDA.plot_magnetic_fields(3)
FDA.plot_log(6,211)
"""

FDAgs=FDA_2D(geo,pde_Maxwels_equation_2D) # gauss - sider iterasyonu sonlu farklar analizi 
# FDAgs.set_fonction_values(FDA.geo.F2D)
# FDAgs.set_more_calculation_area('x0') # x=0 da simetri koşulu dT/dx=0 

# FDAgs.apply_FDA(500)
FDAgs.apply_FDA_with_gauss_sider(500,lamda=0.99)
FDAgs.resoult_FDA.plot_function_values_2D(2,color='-g')
# F2Dgs=FDAgs.resoult_FDA.F2D
Bgs=FDAgs.apply_cross_product()


XX=FDAgs.resoult_FDA.X2D # Kontrol için x değerleri 
YY=FDAgs.resoult_FDA.Y2D # Kontrol için x değerleri 
T_FDA=FDAgs.resoult_FDA.F2D # Sonlu farklar çözümü 

FDAgs.resoult_FDA.plot_stream_line(5,132)
FDAgs.resoult_FDA.plot_magnetic_fields(4)
FDAgs.plot_log(6,212)

FDA=FDAgs

## Buradan sonra Polinom Kollokasyonu Uygulanacak 
Coll_Poly2D=collocation_2D(par=[6,6],Type='poly')

def fcn_boundry(geo,model):  # x=L, y=0, y=L ==> T=0 
    Matrix=model.get_matrix(geo.X,geo.Y,der='f')
    Resoults=geo.F
    return Matrix,Resoults

def fcn_PDE(geo,model): # U=[[U],[dU/dx,dU/dy], [d2U/dx2,d2U/dxdy,d2U/dy2]]
    Ux=model.get_matrix(geo.X,geo.Y,der='fx')
    Uy=model.get_matrix(geo.X,geo.Y,der='fy')
    
    Uxx=model.get_matrix(geo.X,geo.Y,der='fxx')
    Uyy=model.get_matrix(geo.X,geo.Y,der='fyy')
    #Matrix=-(geo.k*model.Ir)*(Uxx+Uyy) # Diferansiyel denklem / Differential equation 
    Matrix=-(geo.dk_dx*model.Ir)*Ux-(geo.dk_dy*model.Ir)*Uy-(geo.k*model.Ir)*(Uxx+Uyy) # Diferansiyel denklem / Differential equation 
    Resoults=geo.q
    return Matrix,Resoults

# Sınır koşullarını belirle 
cond_boundry_x0=lambda x,y: x==0
cond_boundry_xL=lambda x,y: x==L
cond_boundry_y0=lambda x,y: y==0
cond_boundry_yL=lambda x,y: y==L

Coll_Poly2D.add_boundry(FDA.resoult_FDA,fcn_boundry,cond_boundry_x0,color='xb') # x=0 için bütün noktalar 
Coll_Poly2D.add_boundry(FDA.resoult_FDA,fcn_boundry,cond_boundry_xL) # x=L için bütün noktalar 
Coll_Poly2D.add_boundry(FDA.resoult_FDA,fcn_boundry,cond_boundry_y0) # y=0 için bütün noktalar 
Coll_Poly2D.add_boundry(FDA.resoult_FDA,fcn_boundry,cond_boundry_yL) # y=L için bütün noktalar 
print('boundry count',Coll_Poly2D.number_of_boundry_geometry)

# Gövdeleri belirle 
cond_body=lambda x,y: x>0 and x<L and y>0 and y<L
Coll_Poly2D.add_body(FDA.resoult_FDA,fcn_PDE,cond_body) # y=L için bütün noktalar 
print('body count',Coll_Poly2D.number_of_body_geometry)


# Kollokasyon uygulama 
Coll_Poly2D.apply_collocation()

XX=FDA.resoult_FDA.X2D[1:-1,1:-1]
YY=FDA.resoult_FDA.Y2D[1:-1,1:-1]

FDA.resoult_FDA.plot_function_2D(7,221)
FDA.resoult_FDA.plot_stream_line(7,222)

Coll_Poly2D.plot_function(7, 223, XX, YY)
Coll_Poly2D.plot_stream_line(7, 224, XX, YY)
Coll_Poly2D.plot_curl_2D(8, XX, YY)