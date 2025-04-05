from class_Collocation import *
from class_Geometry import *

def exmple_function_for_boundry(geo,U): 
    values=U[0]# fonksiyonun değeri/ the value of the function
    return values
def exmple_function_for_boundry(geo,U): 
    derivative=U[1] # fonksiyonun birinci türevi / first derivative of function 
    return derivative

def exmple_function_for_PDE(geo,U): 
    diferantial_loss=geo.dk*U[0]+geo.k*U[2] # Diferansiyel denklem / Differential equation 
    return diferantial_loss

class PINN_1D:
    def __init__(self,errType='rmse'): # PINN modelini oluştur 
        self.model = tf.keras.Sequential()
        #self.model = SequentialModel(tf.float32)
        self.dtype=self.model.dtype
        self.boundry_points=points_1D()
        self.body_points=points_1D()
        self.list_of_boundry_points=[]
        self.list_of_body_points=[]
        self.number_of_boundry_geometry=0 
        self.number_of_body_geometry=0
        self.err_type=errType
        self.limit_of_error=1e-5
        self.usePoly=False
    
    def add_poly(self,n):
        # F(x)=U0(x)+sum(Ci Ui(x)) şeklinde olması durumunda U0(x) başlangıç fonksiyonunu 
        # polinom olarak tanımla 
        # Dikkat bura tüm sınır şartları fonksiyon ile ilgili olarak kabul ediliyor 
        self.Poly=polinom_collocation_1D(n,self.model.dtype)

    def get_poly(self):# Noktaları kullanarak polinom katsayılarını hesapla 
        self.body_points.convert_tensor(self.model.dtype)
        self.boundry_points.convert_tensor(self.model.dtype)
        self.Poly.add_points(self.boundry_points,'b')
        self.Poly.add_points(self.body_points,'d')
        self.Poly.apply_collocation()

    def add_boundry(self,geo,ids,fcn): # Sınır noktalarda koşulları listeye kaydet 
        x,f,k,dk,q=geo.get_value(ids)
        tmp_boundry=points_1D(fcn)
        if len(x)>0: 
            for i in range(len(x)):
              self.boundry_points.add_point(x[i],f[i],k[i],dk[i],q[i])  # silinecek 
              tmp_boundry.add_point(x[i],f[i],k[i],dk[i],q[i])  
            self.list_of_boundry_points.append(tmp_boundry)
            self.number_of_boundry_geometry+=1
        else: 
            print('veri aktarılırken bir hata oluştu')
        
    def add_body(self,geo,ids,fcn):# Gövde üzerindeki noktaları listeye kaydet 
        x,f,k,dk,q=geo.get_value(ids)
        tmp_body=points_1D(fcn)
        if len(x)>1: 
            for i in range(len(x)):
              self.body_points.add_point(x[i],f[i],k[i],dk[i],q[i])  # silinicek 
              tmp_body.add_point(x[i],f[i],k[i],dk[i],q[i])  
            self.list_of_body_points.append(tmp_body)
            self.number_of_body_geometry+=1
        else: 
            print('veri aktarılırken bir hata oluştu')
 
    def get_boundry_loss(self): # Sınır bölgelerinde hata değerlerini hesapla 
        total_loss=tf.zeros([1,])
        for boundry_points in self.list_of_boundry_points: # 
            with tf.GradientTape(persistent=True) as tape_u:
                tape_u.watch(boundry_points.X)
                U = self.model(boundry_points.X,training=True)
                U_x = tape_u.gradient(U, boundry_points.X)
            # if self.usePoly:
            #     Fp_=self.Poly.predict(boundry_points.X,'f')
            #     Fp_x=self.Poly.predict(boundry_points.X,'fx')
            #     U+=tf.reshape(Fp_, U.shape)
            #     U_x +=tf.reshape(Fp_x, U_x.shape)
            #print('NN:',prediction_of_boundry.shape,'poly:',Fp.shape)
            prediction_of_boundry=boundry_points.fcn(boundry_points,[U,U_x])
            loss_boundry=loss_Func(boundry_points.F,prediction_of_boundry,self.err_type)
            total_loss=total_loss+loss_boundry
        
        return loss_boundry
    
    # def get_boundry_loss_old(self): # Sınır bölgelerinde hata değerlerini hesapla 
        
    #     prediction_of_boundry = self.model(self.boundry_points.X,training=True)
    #     if self.usePoly:
    #         Fp=self.Poly.predict(self.boundry_points.X,'f')
    #         prediction_of_boundry+=tf.reshape(Fp, prediction_of_boundry.shape)
    #         #print('NN:',prediction_of_boundry.shape,'poly:',Fp.shape)
    #     loss_boundry=loss_Func(self.boundry_points.F,prediction_of_boundry,self.err_type)
    #     return loss_boundry
    
    
    def get_body_loss(self): # Diferansiyel hatalarını hesapla 
        total_loss=tf.zeros([1,])
        for body_points in self.list_of_body_points: # 
            with tf.GradientTape(persistent=True) as tape_u:
                tape_u.watch(body_points.X)
                U = self.model(body_points.X,training=True)
                U_x = tape_u.gradient(U, body_points.X)
                U_xx = tape_u.gradient(U_x, body_points.X)
            # if self.usePoly:
            #     Fp_=self.Poly.predict(body_points.X,'f')
            #     Fp_x=self.Poly.predict(body_points.X,'fx')
            #     Fp_xx=self.Poly.predict(body_points.X,'fxx')
            #     U+=tf.reshape(Fp_, U.shape)
            #     U_x +=tf.reshape(Fp_x, U_x.shape)
            #     U_xx +=tf.reshape(Fp_xx, U_xx.shape)
            
            #differantial_equation=body_points.dk*U_x+body_points.k*U_xx+body_points.q
            differantial_equation=body_points.fcn(body_points,[U,U_x,U_xx])
            loss_body=loss_Func(tf.zeros_like(differantial_equation),differantial_equation,self.err_type)
            total_loss=total_loss+loss_body
        return total_loss
    

    def get_random_parameters(self,use=True):# Eğitim aşamasında rasgele parametre eğitmek için kullanılıyor 
        if use:
            random_par = tf.random.uniform(shape=self.model.trainable_variables.shape, 
                                           minval=0, maxval=2, dtype=self.model.dtype)
            random_par=tf.cast(random_par, self.model.dtype)
        else: 
            random_par=tf.ones_like(self.model.trainable_variables)
        return random_par
    
    def train(self,epoch,lr=0.0001,num=100,c=[0.5,0.5],errType='rmse',usePoly=False): # PINN katsayılarını eğit 
        self.usePoly=usePoly
        self.err_type=errType
        for body_points in self.list_of_body_points:
            if not body_points.isTensor:
                body_points.convert_tensor(self.model.dtype)
                
        for boundry_points in self.list_of_boundry_points:
            if not boundry_points.isTensor:
                boundry_points.convert_tensor(self.model.dtype)
                

        optimizer=tf.optimizers.Adam(learning_rate =lr, 
                             beta_1 = 0.75, beta_2 = 0.99, epsilon = 1e-8)
        
        trainable_variables = self.model.trainable_variables
        np_variables = [v.numpy() for v in trainable_variables]
        self.t0=t.time()
        for i in range(epoch):
            with tf.GradientTape(persistent=True) as tape:#
                #tape.watch(trainable_variables) #self.model.trainable_variables[2].value
                loss_boundry=self.get_boundry_loss()
                loss_body=self.get_body_loss()
                loss=c[0]*loss_boundry+c[1]*loss_body
            dW=tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(dW, self.model.trainable_variables))
            #np_variables2 = [v.numpy() for v in trainable_variables]
            if i % num==0 or i<10: 
                print(i,'\n lbound:',loss_boundry.numpy(),
                      '\n lbody:',loss_body.numpy(),'\n loss:',loss.numpy())
                
            if loss.numpy()<self.limit_of_error:
                print('çözüme ulaşıldı l:',loss)
                break 
        self.SIM_time=t.time()-self.t0
        print('PINN simulation time : ',self.SIM_time)
        
    def predict(self,x):
        X=tf.convert_to_tensor(x,dtype=self.model.dtype)
        X=tf.reshape(X,[-1,1])
        prediction= self.model.predict(X)
        # if self.usePoly:
        #     Fp=self.Poly.predict(X,'f')
        #     #print('pred:',prediction,'F:',Fp)
        #     prediction+=np.reshape(Fp.numpy(),prediction.shape)
        #     #print('pred:',prediction)
        return prediction
        
        
class PINN_2D:
    def __init__(self,errType='rmse'): # PINN modelini oluştur 
        self.model = tf.keras.Sequential()
        #self.model = SequentialModel(tf.float32)
        self.dtype=self.model.dtype
        self.boundry_points=points_1D()
        self.body_points=points_1D()
        self.list_of_boundry_points=[]
        self.list_of_body_points=[]
        self.number_of_boundry_geometry=0 
        self.number_of_body_geometry=0
        self.err_type=errType
        self.limit_of_error=1e-5
        self.usePoly=False
    
    def add_poly(self,n):
        # F(x)=U0(x)+sum(Ci Ui(x)) şeklinde olması durumunda U0(x) başlangıç fonksiyonunu 
        # polinom olarak tanımla 
        # Dikkat bura tüm sınır şartları fonksiyon ile ilgili olarak kabul ediliyor 
        self.Poly=polinom_collocation_1D(n,self.model.dtype)

    def get_poly(self):# Noktaları kullanarak polinom katsayılarını hesapla 
        self.body_points.convert_tensor(self.model.dtype)
        self.boundry_points.convert_tensor(self.model.dtype)
        self.Poly.add_points(self.boundry_points,'b')
        self.Poly.add_points(self.body_points,'d')
        self.Poly.apply_collocation()

    def add_boundry(self,geo,fcn,condition,color='xk'): # Sınır noktalarda koşulları listeye kaydet 
        x,f,k,dk,q=geo.get_value(condition)
        tmp_boundry=points_2D(fcn,color)
        tmp_boundry.add_points(x, f, k, dk, q)
        self.list_of_boundry_points.append(tmp_boundry)
        self.number_of_boundry_geometry+=1
        print('sınır/boundry noktaları eklendi')
        # if len(x)>0: 
        #     for i in range(len(x)):
        #       self.boundry_points.add_point(x[i],f[i],k[i],dk[i],q[i])  # silinecek 
        #       tmp_boundry.add_point(x[i],f[i],k[i],dk[i],q[i])  
        #     self.list_of_boundry_points.append(tmp_boundry)
        #     self.number_of_boundry_geometry+=1
        # else: 
        #     print('veri aktarılırken bir hata oluştu')
        
    def add_body(self,geo,fcn,condition,color='.k'):# Gövde üzerindeki noktaları listeye kaydet 
        x,f,k,dk,q=geo.get_value(condition)
        tmp_body=points_2D(fcn,color)
        tmp_body.add_points(x, f, k, dk, q)
        self.list_of_body_points.append(tmp_body)
        self.number_of_body_geometry+=1
        print('govde/body noktaları eklendi')
        # x,f,k,dk,q=geo.get_value(ids)
        # tmp_body=points_1D(fcn)
        # if len(x)>1: 
        #     for i in range(len(x)):
        #       self.body_points.add_point(x[i],f[i],k[i],dk[i],q[i])  # silinicek 
        #       tmp_body.add_point(x[i],f[i],k[i],dk[i],q[i])  
        #     self.list_of_body_points.append(tmp_body)
        #     self.number_of_body_geometry+=1
        # else: 
        #     print('veri aktarılırken bir hata oluştu')
 
    def get_boundry_loss(self): # Sınır bölgelerinde hata değerlerini hesapla 
        total_loss=tf.zeros([1,])
        for boundry_points in self.list_of_boundry_points: # 
            # X=tf.Variable(tf.reshape(bp.X[:,0],[-1,1]))
            # Y=tf.Variable(tf.reshape(bp.X[:,1],[-1,1]))
            with tf.GradientTape(persistent=True) as tape_u:
                r=tf.concat((boundry_points.X,boundry_points.Y), axis=1)
                #tape_u.watch(boundry_points.X)
                U = self.model(r,training=True)
                U_x = tape_u.gradient(U, boundry_points.X)
                U_y = tape_u.gradient(U, boundry_points.Y)
            # if self.usePoly:
            #     Fp_=self.Poly.predict(boundry_points.X,'f')
            #     Fp_x=self.Poly.predict(boundry_points.X,'fx')
            #     U+=tf.reshape(Fp_, U.shape)
            #     U_x +=tf.reshape(Fp_x, U_x.shape)
            #print('NN:',prediction_of_boundry.shape,'poly:',Fp.shape)
            oss_on_boundry=boundry_points.fcn(boundry_points,[[U],[U_x,U_y]])
            #loss_boundry=loss_Func(boundry_points.F,prediction_of_boundry,self.err_type)
            loss_boundry=loss_func_with_err(oss_on_boundry,self.err_type)
            total_loss=total_loss+loss_boundry
            del tape_u
        return total_loss
    
    # def get_boundry_loss_old(self): # Sınır bölgelerinde hata değerlerini hesapla 
        
    #     prediction_of_boundry = self.model(self.boundry_points.X,training=True)
    #     if self.usePoly:
    #         Fp=self.Poly.predict(self.boundry_points.X,'f')
    #         prediction_of_boundry+=tf.reshape(Fp, prediction_of_boundry.shape)
    #         #print('NN:',prediction_of_boundry.shape,'poly:',Fp.shape)
    #     loss_boundry=loss_Func(self.boundry_points.F,prediction_of_boundry,self.err_type)
    #     return loss_boundry
    
    
    def get_body_loss(self): # Diferansiyel hatalarını hesapla 
        total_loss=tf.zeros([1,])
        for body_points in self.list_of_body_points: # 
            # X=tf.Variable(tf.reshape(body_points.X[:,0],[-1,1]))
            # Y=tf.Variable(tf.reshape(body_points.X[:,1],[-1,1]))
            
            with tf.GradientTape(persistent=True) as tape_u:
                #tape_u.watch(X)
                r=tf.concat((body_points.X,body_points.Y), axis=1)
                # tape_u.watch(X)
                # tape_u.watch(Y)
                U = self.model(r,training=True)
                #dU = tape_u.gradient(U, body_X)
                #ddU = tape_u.gradient(dU, body_X)
                U_x = tape_u.gradient(U, body_points.X)
                U_y = tape_u.gradient(U, body_points.Y)
                U_xx = tape_u.gradient(U_x, body_points.X)
                U_xy = tape_u.gradient(U_x, body_points.Y)
                U_yy = tape_u.gradient(U_y, body_points.Y)
                # DU=tape_u.jacobian(U, body_points.X)
            # if self.usePoly:
            #     Fp_=self.Poly.predict(body_points.X,'f')
            #     Fp_x=self.Poly.predict(body_points.X,'fx')
            #     Fp_xx=self.Poly.predict(body_points.X,'fxx')
            #     U+=tf.reshape(Fp_, U.shape)
            #     U_x +=tf.reshape(Fp_x, U_x.shape)
            #     U_xx +=tf.reshape(Fp_xx, U_xx.shape)
            
            #differantial_equation=body_points.dk*U_x+body_points.k*U_xx+body_points.q
            differantial_equation=body_points.fcn(body_points,[[U],[U_x,U_y],[U_xx,U_xy,U_yy]])
            loss_body=loss_Func(tf.zeros_like(differantial_equation),differantial_equation,self.err_type)
            total_loss=total_loss+loss_body
            del tape_u
        return total_loss
    

    def get_random_parameters(self,parameters,use=True):# Eğitim aşamasında rasgele parametre eğitmek için kullanılıyor 
        par_list=[]
        for w in parameters:
            if use:
                random_par = tf.random.uniform(shape=w.shape, 
                                               minval=0, maxval=1.0, dtype=self.model.dtype)
                random_par=tf.cast(random_par, self.model.dtype)
                #random_par=tf.round(random_par,0)
            else: 
                random_par=tf.ones_like(w)
            par_list.append(random_par*w)
        return par_list
    
    def train(self,epoch,lr=0.0001,num=100,c=[0.5,0.5],errType='rmse',usePoly=False): # PINN katsayılarını eğit 
        self.usePoly=usePoly
        self.err_type=errType
        self.log=np.zeros([epoch,4])
        for body_points in self.list_of_body_points:
            if not body_points.isTensor:
                body_points.convert_tensor(self.model.dtype)
                
        for boundry_points in self.list_of_boundry_points:
            if not boundry_points.isTensor:
                boundry_points.convert_tensor(self.model.dtype)
                

        optimizer=tf.optimizers.Adam(learning_rate =lr, 
                             beta_1 = 0.75, beta_2 = 0.999, epsilon = 1e-8)
        
        trainable_variables = self.model.trainable_variables
        np_variables = [v.numpy() for v in trainable_variables]
        self.t0=t.time()
        for i in range(epoch):
            with tf.GradientTape(persistent=True) as tape:#
                #tape.watch(trainable_variables) #self.model.trainable_variables[2].value
                loss_boundry=self.get_boundry_loss()
                loss_body=self.get_body_loss()
                loss=c[0]*loss_boundry+c[1]*loss_body
            dW=tape.gradient(loss, self.model.trainable_variables)
            ddW=self.get_random_parameters(dW)
            optimizer.apply_gradients(zip(ddW, self.model.trainable_variables))
            #np_variables2 = [v.numpy() for v in trainable_variables]
            self.log[i,:]=[i,loss_boundry[0].numpy(),loss_body[0].numpy(),loss[0].numpy()]
            if i % num==0 or i<10: 
                print(i,'\n lbound:',loss_boundry.numpy(),
                      '\n lbody:',loss_body.numpy(),'\n loss:',loss.numpy())
                
            if loss.numpy()<self.limit_of_error:
                print('çözüme ulaşıldı l:',loss)
                break 
        self.SIM_time=t.time()-self.t0
        print('PINN simulation time : ',self.SIM_time)
        
    def predict(self,x,y):
        shape_X=x.shape 
        np_x=np.reshape(x,[-1,1])
        np_y=np.reshape(y,[-1,1])
        np_X=np.concatenate((np_x, np_y),axis=1)
        X=tf.convert_to_tensor(np_X,dtype=self.model.dtype)
        prediction= self.model.predict(X)
        # if self.usePoly:
        #     Fp=self.Poly.predict(X,'f')
        #     #print('pred:',prediction,'F:',Fp)
        #     prediction+=np.reshape(Fp.numpy(),prediction.shape)
        #     #print('pred:',prediction)
        prediction2D=np.reshape(prediction,shape_X)
        return prediction2D
    
    def apply_curl(self,np_x,np_y):
        shapeX=np_x.shape 
        np_X=np.reshape(np_x,[-1,1])
        np_Y=np.reshape(np_y,[-1,1])
        X=tf.convert_to_tensor(np_X)
        Y=tf.convert_to_tensor(np_Y)
        X=tf.Variable(X)
        Y=tf.Variable(Y)
        with tf.GradientTape(persistent=True) as tape_u:
            r=tf.concat((X,Y), axis=1)
            U = self.model(r,training=True)
            U_x = tape_u.gradient(U, X)
            U_y = tape_u.gradient(U, Y)
            U_xx = tape_u.gradient(U_x, X)
            U_xy = tape_u.gradient(U_x, Y)
            U_yy = tape_u.gradient(U_y, Y)
        np_vector=[np.reshape(U_y.numpy(),shapeX),
                   np.reshape(-U_x.numpy(),shapeX),
                   np.reshape(tf.sqrt((U_x**2+U_y**2)).numpy(),shapeX)]
        return np_vector
    
    def plot_log(self,figNo): 
        plot_points(figNo,311,self.log[:,0],self.log[:,3],title='total loss')
        plot_points(figNo,312,self.log[:,0],self.log[:,1],title='boundry loss')
        plot_points(figNo,313,self.log[:,0],self.log[:,2],title='body loss')
        
    def plot_stream_line(self,figNo,subNo,X2D,Y2D,title='stream line'):
        B=self.apply_curl(X2D,Y2D)
        plot_stream_lines(figNo,subNo,X2D,Y2D,B[0],B[1],title)
        
    def plot_points_values_2D(self,figNo,subNo, color=None):
        plot_list=[]
        for bp in self.list_of_boundry_points: 
            plot_list.append((bp.np_x,bp.np_y,bp.color))
        for bp in self.list_of_body_points: 
            plot_list.append((bp.np_x,bp.np_y,bp.color))
            #plot_points(figNo,221,bp.np_x[:,0],bp.np_x[:,1],'xb','noktalar')
        # plot_surf_2D(figNo,222,self.X2D,self.Y2D,self.F2D-np.ones_like(self.F2D),'funksiyon')
        # plot_surf_2D(figNo,223,self.X2D,self.Y2D,self.par2D,'parametre')
        # plot_surf_2D(figNo,224,self.X2D,self.Y2D,self.q2D,'ısı üretimi')
        plot_list_points(figNo,subNo,plot_list,'noktalar')