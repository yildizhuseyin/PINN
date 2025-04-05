# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 00:07:33 2025

@author: HY
"""
from class_Geometry import *


class Poly_2D: 
    x,y,n,m= symbols('x y n m')
    e=2.718281828459045;
    Type=None
    def __init__(self,n=[2,2],dtype=tf.float32): # Baz fonksiyonunu tanımla 
        self.Type='poly'
        self.nn=n;
        self.dtype=dtype
        self.get_ustel_vectors(n)
        self.Poly=self.get_poly(self.x,self.y,self.n,self.m)
        self.P=lambdify([self.x,self.y], self.Poly, "math")
        self.Px=lambdify([self.x,self.y], diff(self.Poly,self.x), "math")
        self.Py=lambdify([self.x,self.y], diff(self.Poly,self.y), "math")
        self.Pxx=lambdify([self.x,self.y], diff(self.Poly,self.x,self.x), "math")
        self.Pxy=lambdify([self.x,self.y], diff(self.Poly,self.x,self.y), "math")
        self.Pyy=lambdify([self.x,self.y], diff(self.Poly,self.y,self.y), "math")
        
        self.np_P=lambdify([self.x,self.y,self.n,self.m], self.Poly, "numpy")
        self.np_Px=lambdify([self.x,self.y,self.n,self.m], diff(self.Poly,self.x), "numpy")
        self.np_Py=lambdify([self.x,self.y,self.n,self.m], diff(self.Poly,self.y), "numpy")
        self.np_Pxy=lambdify([self.x,self.y,self.n,self.m], diff(self.Poly,self.x,self.y), "numpy")
        self.np_Pxx=lambdify([self.x,self.y,self.n,self.m], diff(self.Poly,self.x,self.x), "numpy")
        self.np_Pyy=lambdify([self.x,self.y,self.n,self.m], diff(self.Poly,self.y,self.y), "numpy")
        
        self.tf_P=lambdify([self.x,self.y,self.n,self.m], self.Poly, "tensorflow")
        self.tf_Px=lambdify([self.x,self.y,self.n,self.m], diff(self.Poly,self.x), "tensorflow")
        self.tf_Py=lambdify([self.x,self.y,self.n,self.m], diff(self.Poly,self.y), "tensorflow")
        self.tf_Pxy=lambdify([self.x,self.y,self.n,self.m], diff(self.Poly,self.x,self.y), "tensorflow")
        self.tf_Pxx=lambdify([self.x,self.y,self.n,self.m], diff(self.Poly,self.x,self.x), "tensorflow")
        self.tf_Pyy=lambdify([self.x,self.y,self.n,self.m], diff(self.Poly,self.y,self.y), "tensorflow")
        
    def get_poly(self,x,y,nx,ny):
        f=Pow(x,nx)*Pow(y,ny);
        print(x,y,nx,ny,f)
        return f
    
    def get_ustel_vectors(self,n):
        nn=int(n[0]*n[1])
        np_M=np.zeros([1,nn],dtype=np.int32);
        np_N=np.zeros([1,nn],dtype=np.int32);
        np_Ir=np.ones([1,nn]);
        say=0
        for i in range(n[0]): 
            for j in range(n[1]):
                np_M[0,say]=i
                np_N[0,say]=j
                say=say+1
        self.M=tf.convert_to_tensor(np_M,dtype=self.dtype);
        self.N=tf.convert_to_tensor(np_N,dtype=self.dtype);
        self.Ir=tf.convert_to_tensor(np_Ir,dtype=self.dtype);
        self.C=tf.zeros_like(self.M)
        self.C=tf.Variable(self.C)
        self.trainable_variables=self.C
                
    def get_matrix(self,x,y,der='f'):
        x=np.reshape(x, [-1,1])
        y=np.reshape(y, [-1,1])
        self.N=int(self.nn[0]*self.nn[1])
        M=np.zeros([len(x[:,0]),self.N]);
        say=0;
        for i in range(self.nn[0]): 
            for j in range(self.nn[1]):
                if der=='f':
                    par=self.tf_P(x,y,j,i)
                elif der=='fx':
                    par=self.tf_Px(x,y,j,i)
                elif der=='fy':
                    par=self.tf_Py(x,y,j,i)
                elif der=='fxy':
                    par=self.tf_Pxy(x,y,j,i)
                elif der=='fxx':
                    par=self.tf_Pxx(x,y,j,i)
                elif der=='fyy':
                    par=self.tf_Pyy(x,y,j,i)
                M[:,say]=par[:,0]
                say=say+1
        return M
    
    def model(self,x,y,der='f'):
        M=self.get_matrix(x,y,der)
        F=tf.matmul(M, self.C)
        return F 
    
    def predict(self,x,y,der='f'):
        shapeX=x.shape
        np_X=np.reshape(x,[-1,1])
        np_Y=np.reshape(y,[-1,1])
        X=tf.convert_to_tensor(np_X)
        Y=tf.convert_to_tensor(np_Y)
        F=self.model(X,Y,der)
        np_f=np.reshape(F.numpy(),shapeX)
        return np_f

    
class RBF_function_2D:

    x,y,xk,yk,ex,ey= symbols('x y xk yk ex ey')
    e=2.718281828459045;
    Type=None
    def __init__(self,Type): # Baz fonksiyonunu tanımla 
        self.Type=Type
        if Type=='gauss':
            F=self.rbf_gauss(self.x, self.y, self.xk, self.yk, self.ex, self.ey)
        elif Type=='mquad':
            F=self.rbf_MultiQuadric(self.x, self.y, self.xk, self.yk, self.ex, self.ey) 
        elif Type=='iquad':
            F=self.rbf_InverseQuadratic(self.x, self.y, self.xk, self.yk, self.ex, self.ey) 
        elif Type=='imquad':
            F=self.rbf_InverseMultiQuadric(self.x, self.y, self.xk, self.yk, self.ex, self.ey)    
        elif Type=='eUzal':
            F=self.rbf_eUzal(self.x, self.y, self.xk, self.yk, self.ex, self.ey)

            
        self.Fcn=F    
        print(self.Fcn)        
        self.tf_F=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], self.Fcn, "tensorflow")
        self.tf_Fx=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], diff(self.Fcn,self.x), "tensorflow")
        self.tf_Fy=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], diff(self.Fcn,self.y), "tensorflow")
        self.tf_Fxx=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], diff(self.Fcn,self.x,self.x), "tensorflow")
        self.tf_Fxy=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], diff(self.Fcn,self.x,self.y), "tensorflow")
        self.tf_Fyy=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], diff(self.Fcn,self.y,self.y), "tensorflow")
            
    
    def rbf_gauss(self, x, y, xm, ym, ex, ey):
       
        Fx=exp(-(ex*ex)*((x-xm)**2)) 
        Fy=exp(-(ey*ey)*((y-ym)**2)) 
        return Fx*Fy
    
    def rbf_MultiQuadric(self, x, y, xm, ym, ex, ey):
        Fx=((1+(ex*ex)*(x-xm))**2)**0.5
        Fy=((1+(ey*ey)*(y-ym))**2)**0.5
        return Fx*Fy
    
    def rbf_InverseQuadratic(self, x, y, xm, ym, ex, ey):    
        Fx=(1+(ex*ex)*(x-xm))**2
        Fy=(1+(ey*ey)*(y-ym))**2
        return 1/Fx*Fy

    def rbf_InverseMultiQuadric(self, x, y, xm, ym, ex, ey):# Inverse multiquadric formunda baz fonksiyon 
        Fx=((1+(ex*ex)*(x-xm))**2)**0.5
        Fy=((1+(ey*ey)*(y-ym))**2)**0.5
        return 1/Fx*Fy

    def rbf_eUzal(self, x, y, xm, ym, ex, ey):#Erol Hocanın önerdiği formda baz fonksiyon  
        Fx = 1+(x-xm)**2
        Fy = 1+(y-ym)**2
        return 1/Fx*Fy
        
def fcn_boundry_example(geo,model):  # x=L, y=0, y=L ==> T=0 
    Matrix=model.get_matrix(geo.X,geo.Y,der='f')
    Resoults=geo.F
    return Matrix,Resoults

def fcn_PDE_example(geo,model): # U=[[U],[dU/dx,dU/dy], [d2U/dx2,d2U/dxdy,d2U/dy2]]
    Uxx=model.get_matrix(geo.X,geo.Y,der='fxx')
    Uyy=model.get_matrix(geo.X,geo.Y,der='fyy')
    Matrix=(geo.k*model.Ir)*(Uxx+Uyy) # Diferansiyel denklem / Differential equation 
    Resoults=geo.q
    return Matrix,Resoults

class collocation_2D: 
    x,y,n,m= symbols('x y n m')
    e=2.718281828459045;
    Type=None
    def __init__(self,par,Type='poly',errType='rmse',dtype=tf.float64): # Baz fonksiyonunu tanımla 
        self.Type=Type
        if self.Type=='poly':
            self.FCN=Poly_2D(n=par,dtype=dtype)
        elif self.Type=='rbf': 
            self.FCN=RBF_function_2D()
        
        self.dtype=dtype
        self.list_of_boundry_geometry=[]
        self.list_of_body_geometry=[]
        self.number_of_boundry_geometry=0 
        self.number_of_body_geometry=0
        self.err_type=errType
        self.limit_of_error=1e-5
        #self.usePoly=False

    def add_boundry(self,geo,fcn,condition,color='xk'): # Sınır noktalarda koşulları listeye kaydet 
        x,f,k,dk,q=geo.get_value(condition)
        tmp_boundry=points_2D(fcn,color)
        tmp_boundry.add_points(x, f, k, dk, q)
        self.list_of_boundry_geometry.append(tmp_boundry)
        self.number_of_boundry_geometry+=1
        print('sınır/boundry noktaları eklendi')
    
    def add_body(self,geo,fcn,condition,color='.k'):# Gövde üzerindeki noktaları listeye kaydet 
        x,f,k,dk,q=geo.get_value(condition)
        tmp_body=points_2D(fcn,color)
        tmp_body.add_points(x, f, k, dk, q)
        self.list_of_body_geometry.append(tmp_body)
        self.number_of_body_geometry+=1
        print('govde/body noktaları eklendi')
    
    
    def apply_collocation(self):
        say=0; 
        for bp in self.list_of_boundry_geometry:
            if not bp.isTensor:
                bp.convert_tensor(self.dtype)
            if say==0 :
                M,B=bp.fcn_m(bp,self.FCN)
            else: 
                tmpM,tmpB=bp.fcn_m(bp,self.FCN)
                M=tf.concat((M,tmpM),axis=0)
                B=tf.concat((B,tmpB),axis=0)
            say+=1 
    
        for bp in self.list_of_body_geometry:
            if not bp.isTensor:
                bp.convert_tensor(self.dtype)
            if say==0 :
                M,B=bp.fcn_m(bp,self.FCN)
            else: 
                tmpM,tmpB=bp.fcn_m(bp,self.FCN)
                M=tf.concat((M,tmpM),axis=0)
                B=tf.concat((B,tmpB),axis=0)
            say+=1 
        tmpC=tf_linsolve(M,B)
        self.FCN.C=tmpC
        
    def predict(self,x,y):
        prediction= self.FCN.predict(np_X,np_Y)
        prediction2D=np.reshape(prediction,shape_X)
        return prediction2D
    
    def apply_curl(self,np_x,np_y):
        U_x = self.FCN.predict(np_x,np_y,'fx')
        U_y = self.FCN.predict(np_x,np_y,'fy')
        np_vector=[U_y,
                   -U_x,
                   np.sqrt(U_x**2+U_y**2)]
        return np_vector
    
    def plot_function(self,figNo,subNo,X2D,Y2D,title='collocation'):
        resoults=self.FCN.predict(X2D,Y2D)
        plot_surf_2D(figNo,subNo,X2D,Y2D,resoults,title)

    def plot_stream_line(self,figNo,subNo,X2D,Y2D,title='stream line'):
        B=self.apply_curl(X2D,Y2D)
        plot_stream_lines(figNo,subNo,X2D,Y2D,B[0],B[1],title)
        
    def plot_points_values_2D(self,figNo,subNo, color=None):
        plot_list=[]
        for bp in self.list_of_boundry_geometry: 
            plot_list.append((bp.np_x,bp.np_y,bp.color))
        for bp in self.list_of_body_geometry: 
            plot_list.append((bp.np_x,bp.np_y,bp.color))

        plot_list_points(figNo,subNo,plot_list,'noktalar')
    
    def plot_curl_2D(self,figNo,x2D,y2D,color=None):
        B=self.apply_curl(x2D,y2D)
        self.plot_points_values_2D(figNo,224)
        plot_surf_2D(figNo,223,x2D,y2D,B[2],'funksiyon')
        plot_surf_2D(figNo,221,x2D,y2D,B[0],'Bx')
        plot_surf_2D(figNo,222,x2D,y2D,B[1],'By')