# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 00:07:33 2025

@author: HY
"""
from class_Geometry import *

import cv2

class Poly_2D: 
    x,y,n,m= symbols('x y n m')
    e=2.718281828459045;
    Type=None
    def __init__(self,n=[2,2],dtype=tf.float64): # Baz fonksiyonunu tanımla 
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
    def __init__(self,dtype=tf.float64): # Baz fonksiyonunu tanımla 
        self.color='or'
        self.number_of_center_points=0;
        self.set_function_type()
        self.dtype=dtype
        self.trainable_variables=[]
        
    def set_function_type(self,Type='gauss'):
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
        print(self.Type,'\n',self.Fcn)        
        self.tf_P=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], self.Fcn, "tensorflow")
        self.tf_Px=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], diff(self.Fcn,self.x), "tensorflow")
        self.tf_Py=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], diff(self.Fcn,self.y), "tensorflow")
        self.tf_Pxx=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], diff(self.Fcn,self.x,self.x), "tensorflow")
        self.tf_Pxy=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], diff(self.Fcn,self.x,self.y), "tensorflow")
        self.tf_Pyy=lambdify([self.x,self.y,self.xk,self.yk,self.ex,self.ey], diff(self.Fcn,self.y,self.y), "tensorflow")
        
        
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
        return 1/(Fx*Fy)

    def rbf_InverseMultiQuadric(self, x, y, xm, ym, ex, ey):# Inverse multiquadric formunda baz fonksiyon 
        Fx=((1+(ex*ex)*(x-xm))**2)**0.5
        Fy=((1+(ey*ey)*(y-ym))**2)**0.5
        return 1/(Fx*Fy)

    def rbf_eUzal(self, x, y, xm, ym, ex, ey):#Erol Hocanın önerdiği formda baz fonksiyon  
        Fx = 1+(ex*ex)*(x-xm)**2
        Fy = 1+(ey*ey)*(y-ym)**2
        return 1/(Fx*Fy)
    

    
    def add_center_points(self,xm,ym,ex,ey,c=tf.zeros([1,1])):
        c=tf.cast(c,dtype=self.dtype)
        if self.number_of_center_points==0: 
            self.Xm=tf.reshape(xm,[1,-1]);         self.Ym=tf.reshape(ym,[1,-1])
            self.Ex=tf.reshape(ex,[1,-1]);         self.Ey=tf.reshape(ey,[1,-1]);
            self.C=tf.reshape(c,[-1,1]);
            
        else:
            self.Xm=tf.concat((self.Xm,tf.reshape(xm,[1,-1])),axis=1)
            self.Ym=tf.concat((self.Ym,tf.reshape(ym,[1,-1])),axis=1)
            self.Ex=tf.concat((self.Ex,tf.reshape(ex,[1,-1])),axis=1)
            self.Ey=tf.concat((self.Ey,tf.reshape(ey,[1,-1])),axis=1)
            self.C=tf.concat((self.C,tf.reshape(c,[-1,1])),axis=0)
        self.Xm=tf.cast(self.Xm,dtype=self.dtype)
        self.Ym=tf.cast(self.Ym,dtype=self.dtype)
        self.Ex=tf.cast(self.Ex,dtype=self.dtype)
        self.Ey=tf.cast(self.Ey,dtype=self.dtype)
        self.C=tf.cast(self.C,dtype=self.dtype)

        self.Ir=tf.ones_like(self.Xm)
        self.number_of_center_points=len(self.Xm[0,:].numpy())
                
    def get_matrix(self,x,y,der='f'):
        # x=tf.reshape(x, [-1,1])
        # y=tf.reshape(y, [-1,1])
        IIc=np.ones_like(x.numpy())
        Ic=tf.convert_to_tensor(IIc)
        X=x*self.Ir;          Y=y*self.Ir
        Xm=Ic*self.Xm;        Ym=Ic*self.Ym
        Ex=Ic*self.Ex;        Ey=Ic*self.Ey
        if der=='f':
            M=self.tf_P(X,Y,Xm,Ym,Ex,Ey)
        elif der=='fx':
            M=self.tf_Px(X,Y,Xm,Ym,Ex,Ey)
        elif der=='fy':
            M=self.tf_Py(X,Y,Xm,Ym,Ex,Ey)
        elif der=='fxy':
            M=self.tf_Pxy(X,Y,Xm,Ym,Ex,Ey)
        elif der=='fxx':
            M=self.tf_Pxx(X,Y,Xm,Ym,Ex,Ey)
        elif der=='fyy':
            M=self.tf_Pyy(X,Y,Xm,Ym,Ex,Ey)
        return M
    
    def set_tranible_parameters(self,Xm=False,Ym=False,Ex=False,Ey=False,C=True):
        self.trainable_variables=[]
        if Xm:
            self.Xm=tf.Variable(self.Xm); self.trainable_variables.append(self.Xm)
        if Ym:
            self.Ym=tf.Variable(self.Ym); self.trainable_variables.append(self.Ym)
        if Ex:
            self.Ex=tf.Variable(self.Ex); self.trainable_variables.append(self.Ex)
        if Ey:
            self.Ey=tf.Variable(self.Ey); self.trainable_variables.append(self.Ey)
        if C:
            self.C=tf.Variable(self.C); self.trainable_variables.append(self.C)
            
    
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
            self.FCN=RBF_function_2D(dtype=dtype)
        
        self.dtype=dtype
        self.list_of_boundry_geometry=[]
        self.list_of_body_geometry=[]
        self.list_of_jump_geometry=[]
        self.number_of_boundry_geometry=0 
        self.number_of_body_geometry=0
        self.number_of_jump_geometry=0

        self.err_type=errType
        self.limit_of_error=1e-12
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
        
    def add_jump_geometry(self,geo,fcn,condition,par,n,color='.k'):# Gövde üzerindeki noktaları listeye kaydet 
        x,f,k,dk,q=geo.get_value(condition)
        tmp_jump=points_2D(fcn,color)
        tmp_jump.add_points(x, f, k, dk, q,par=par,nn=n)
        self.list_of_jump_geometry.append(tmp_jump)
        self.number_of_jump_geometry+=1
        print('govde/jump noktaları eklendi')
    
    def RBF_add_center_point_with_points(self,xm,ym,ex,ey):# Gövde üzerindeki noktaları merkez nokta olarak ekle 
        if self.Type=='rbf':    
            self.FCN.add_center_points(xm, ym, ex, ey)
            print('RBF model / merkez noktaları eklendi')
        
    
    def RBF_add_center_point_with_geo(self,geo,condition,par=1.0):# Gövde üzerindeki noktaları merkez nokta olarak ekle 
        xm,ym,dx,dy=geo.get_geometric_value(condition)
        self.FCN.add_center_points(xm, ym, par/dx, par/dy)
        print('RBF model / merkez noktaları eklendi')
        
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
        
        for jp in self.list_of_jump_geometry:
            if not jp.isTensor:
                jp.convert_tensor(self.dtype)
            if say==0 :
                M,B=jp.fcn_m(jp,self.FCN)
            else: 
                tmpM,tmpB=jp.fcn_m(jp,self.FCN)
                M=1*tf.concat((M,tmpM),axis=0)
                B=1*tf.concat((B,tmpB),axis=0)
            say+=1 
        tmpC=tf_linsolve(M,B)
        self.FCN.C=tmpC
        
    def get_boundry_loss(self): # Sınır bölgelerinde hata değerlerini hesapla 
        total_loss=tf.zeros([1,],dtype=self.FCN.dtype)
        for boundry_points in self.list_of_boundry_geometry: # 
            tmpM,tmpB=boundry_points.fcn_m(boundry_points,self.FCN)
            preditction=tf.matmul(tmpM, self.FCN.C)
            loss_boundry=loss_Func(preditction,tmpB,self.err_type)
            #loss_boundry=loss_func_with_err(oss_on_boundry,self.err_type)
            total_loss=total_loss+loss_boundry
        return total_loss
    
    
    def get_body_loss(self): # Diferansiyel hatalarını hesapla 
        total_loss=tf.zeros([1,],dtype=self.FCN.dtype)
        for body_points in self.list_of_body_geometry: # 
            tmpM,tmpB=body_points.fcn_m(body_points,self.FCN)
            preditction=tf.matmul(tmpM, self.FCN.C)
            loss_body=loss_Func(preditction,tmpB,self.err_type)
            total_loss=total_loss+loss_body
        return total_loss
    
    def get_jump_loss(self): # Diferansiyel hatalarını hesapla 
        total_loss=tf.zeros([1,],dtype=self.FCN.dtype)
        for jump_points in self.list_of_jump_geometry: # 
            tmpM,tmpB=jump_points.fcn_m(jump_points,self.FCN)
            preditction=tf.matmul(tmpM, self.FCN.C)
            loss_jump=loss_Func(preditction,tmpB,self.err_type)
            total_loss=total_loss+loss_jump
        return total_loss
        
    def get_random_parameters(self,parameters,use=True):# Eğitim aşamasında rasgele parametre eğitmek için kullanılıyor 
        par_list=[]
        for w in parameters:
            if use:
                random_par = tf.random.uniform(shape=w.shape, 
                                               minval=0, maxval=1.0, dtype=self.FCN.dtype)
                random_par=tf.cast(random_par, self.FCN.dtype)
                #random_par=tf.round(random_par,0)
            else: 
                random_par=tf.ones_like(w)
            par_list.append(random_par*w)
        return par_list
    
    def train(self,epoch,lr=0.0001,num=100,c=[0.25,0.25,0.5],errType='rmse',usePoly=False,reset=True,title='training'): # PINN katsayılarını eğit 
        self.usePoly=usePoly
        self.err_type=errType
        
        log=np.zeros([epoch,4])
        if reset: 
            self.log=np.zeros([1,4])
            
        for body_points in self.list_of_body_geometry:
            if not body_points.isTensor:
                body_points.convert_tensor(self.FCN.dtype)
                
        for boundry_points in self.list_of_boundry_geometry:
            if not boundry_points.isTensor:
                boundry_points.convert_tensor(self.FCN.dtype)
            
        for jump_points in self.list_of_jump_geometry:
            if not jump_points.isTensor:
                jump_points.convert_tensor(self.FCN.dtype)
                
        optimizer=tf.optimizers.Adam(learning_rate =lr, 
                             beta_1 = 0.75, beta_2 = 0.999, epsilon = 1e-8)
        
        trainable_variables = self.FCN.trainable_variables
        np_variables = [v.numpy() for v in trainable_variables]
        self.t0=t.time()
        for i in range(epoch):
            with tf.GradientTape(persistent=True) as tape:#
                #tape.watch(trainable_variables) #self.model.trainable_variables[2].value
                loss_boundry=self.get_boundry_loss()
                loss_body=self.get_body_loss()
                loss_jump=self.get_jump_loss()
                loss=c[0]*loss_boundry+c[1]*loss_body+c[2]*loss_jump
            dW=tape.gradient(loss, self.FCN.trainable_variables)
            ddW=self.get_random_parameters(dW)
            optimizer.apply_gradients(zip(ddW, self.FCN.trainable_variables))
            #np_variables2 = [v.numpy() for v in trainable_variables]
            log[i,:]=[i+self.log[-1,0],loss_boundry[0].numpy(),loss_body[0].numpy(),loss[0].numpy()]
            if i % num==0 or i<10: 
                print('itr:',i+self.log[-1,0],'  sub itr:',i,'  ',title,'\n lbound:',loss_boundry.numpy(),
                      '\n lbody:',loss_body.numpy(),
                      '\n ljump:',loss_jump.numpy(),
                      '\n loss:',loss.numpy(),'\n \n')
                
            if loss.numpy()<self.limit_of_error:
                print('çözüme ulaşıldı l:',loss)
                break 
        self.SIM_time=t.time()-self.t0
        if reset: 
            self.log=log
        else: 
            self.log=np.concatenate((self.log, log))
        print('PINN simulation time : ',self.SIM_time)
        
        del(optimizer)
    
    
        
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
    
    def apply_filter_with_np(self,Data,n):
        sy,sx=Data.shape 
        NewData=Data.copy()
        for i in range(n+1,sx-n-1):
            for j in range(n+1,sy-n-1):
                NewData[j,i]=np.mean(np.mean(Data[j-n:j+n,i-n:i+n]))
        return NewData
    
    def find_max_errs_np(self,Data,val=0.75):

        sy,sx=Data.shape 
        NewData=np.abs(np.reshape(Data.copy(),[-1,]))
        data_max=np.max(np.max(NewData))
        
        for i in range(len(NewData)): 
            if NewData[i]<=data_max*val:
                NewData[i]=0; 
            else: 
                data=1 
        newData2D=np.reshape(NewData,Data.shape)
        norm_data=np.array(255*np.abs(newData2D.copy())/data_max, dtype=np.uint8)
        ret, bw = cv2.threshold(norm_data, 0, 50,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        connectivity = 3
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(norm_data, connectivity, cv2.CV_32S)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        min_size = 1 #threshhold value for objects in scene
        s1,s2=bw.shape
        img2 = np.zeros(([s1,s2,3]), np.uint8)
        params=[]

        if nb_components>1:
            maxVal=np.max(stats[1:-1,4])
        
            for i in range(1, nb_components):
                # use if sizes[i] >= min_size: to identify your objects
                color = np.random.randint(255,size=5)
                # draw the bounding rectangele around each object
                cv2.rectangle(img2, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
                #img2[output == i + 1] = color
                if maxVal==stats[i,4] and nb_components>1:
                    xm=centroids[i,0]/s2;   ym=centroids[i,1]/s1
                    dx=stats[i,2]/s2;       dy=stats[i,3]/s1
                    params.append([[xm,ym],[dx,dy]])
                    self.RBF_add_center_point_with_points(xm,ym,2/dx,2/dy)
        #cv2.imshow("bw",norm_data)
        #cv2.imshow("thresh",img2)
        return newData2D,params
            
    def error_analysis_on_geometry(self,geo,fcn_m,figNo):
        condition=lambda x,y: 1>0

        X2D=geo.X2D
        Y2D=geo.Y2D
        shape2D=X2D.shape
        x,f,k,dk,q=geo.get_value(condition)
        tmp_body=points_2D(fcn_m,'.k')
        tmp_body.add_points(x, f, k, dk, q)
        tmp_body.convert_tensor(self.FCN.dtype)
        tmpM,tmpB=fcn_m(tmp_body,self.FCN)
        preditction=tf.matmul(tmpM, self.FCN.C)
        Ub=self.FCN.get_matrix(tmp_body.X,tmp_body.Y,der='f')
        
        bases_f=tf.matmul(Ub, tf.ones_like(self.FCN.C))
        bases_diff=tf.matmul(tmpM, tf.ones_like(self.FCN.C))
        err=tmpB-preditction
        err_2D=tf.reshape(err,shape2D)
        bases_f_2D=tf.reshape(bases_f,shape2D)
        bases_diff_2D=tf.reshape(bases_diff,shape2D)
        
        filtered=self.apply_filter_with_np(err_2D.numpy(),5)
        filtered2,params=self.find_max_errs_np(filtered)
        #filtered=cv2.GaussianBlur(err_2D.numpy(), (4,4), 0)
        
        plot_surf_2D(figNo,221,X2D[1:-1,1:-1],Y2D[1:-1,1:-1],err_2D[1:-1,1:-1],'err2D')
        plot_surf_3D(figNo,222,X2D[1:-1,1:-1],Y2D[1:-1,1:-1],filtered[1:-1,1:-1],'err3D')
        plot_surf_3D(figNo,223,X2D[1:-1,1:-1],Y2D[1:-1,1:-1],bases_f_2D[1:-1,1:-1],'bases')
        plot_surf_3D(figNo,224,X2D[1:-1,1:-1],Y2D[1:-1,1:-1],filtered2[1:-1,1:-1],'|err3D|')
        #plot_surf_3D(figNo,224,X2D[1:-1,1:-1],Y2D[1:-1,1:-1],abs(err_2D[1:-1,1:-1]),'|err3D|')
        print(params)
            

    def plot_log(self,figNo): 
        plot_points(figNo,311,self.log[:,0],self.log[:,3],title='total loss')
        plot_points(figNo,312,self.log[:,0],self.log[:,1],title='boundry loss')
        plot_points(figNo,313,self.log[:,0],self.log[:,2],title='body loss')
        
    def plot_function(self,figNo,subNo,X2D,Y2D,title='collocation'):
        resoults=self.FCN.predict(X2D,Y2D)
        plot_surf_2D(figNo,subNo,X2D,Y2D,resoults,title)

    def plot_stream_line(self,figNo,subNo,X2D,Y2D,title='stream line'):
        B=self.apply_curl(X2D,Y2D)
        plot_stream_lines(figNo,subNo,X2D,Y2D,B[0],B[1],title)
        
    def plot_points_values_2D(self,figNo,subNo, color=None):
        plot_list=[]
        if self.Type=='rbf':
            plot_list.append((self.FCN.Xm.numpy(),self.FCN.Ym.numpy(),self.FCN.color))
        for bp in self.list_of_boundry_geometry: 
            plot_list.append((bp.np_x,bp.np_y,bp.color))
        for bp in self.list_of_body_geometry: 
            plot_list.append((bp.np_x,bp.np_y,bp.color))
        for jp in self.list_of_jump_geometry: 
            plot_list.append((jp.np_x,jp.np_y,jp.color))
        plot_list_points(figNo,subNo,plot_list,'noktalar')
    
    def plot_curl_2D(self,figNo,x2D,y2D,color=None):
        B=self.apply_curl(x2D,y2D)
        self.plot_points_values_2D(figNo,224)
        plot_surf_2D(figNo,223,x2D,y2D,B[2],'|B|')
        plot_surf_2D(figNo,221,x2D,y2D,B[0],'Bx')
        plot_surf_2D(figNo,222,x2D,y2D,B[1],'By')