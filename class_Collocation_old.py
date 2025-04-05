from class_Geometry import *

        
class polinom_collocation_1D:
    x,p= symbols('x p')
    def __init__(self,n,dType):
        self.n=n+1
        self.dtype=dType
        self.Fcn=self.x**self.p
        self.isBoundry=False
        self.isBody=False
        self.par=1;
        self.r=tf.Variable(range(n),dtype=dType)
        self.Ir=tf.ones_like(self.r)
        self.C=tf.Variable(tf.zeros_like(self.r),dtype=dType)
        self.tf_F=lambdify([self.x,self.p], self.Fcn, "tensorflow")
        self.tf_Fx=lambdify([self.x,self.p], diff(self.Fcn,self.x), "tensorflow")
        self.tf_Fxx=lambdify([self.x,self.p], diff(self.Fcn,self.x,self.x), "tensorflow")
        print('n:',self.n,'r:',self.r.numpy())
        
    def add_points(self,point,Type='d'):
        if Type=='b':
            self.boundry_points=point
            self.isBoundry=True
        elif Type=='d':
            self.body_points=point
            self.isBody=True
        else: 
            print('Hatalı nokta tanımlama...')
            
    def add_externap_par(self,par):
        self.par=par        
    
    def apply_collocation(self):
        Mb=[];  Bb=[]
        Md=[];  Bd=[]
        M=[];  B=[]
        if self.isBoundry: 
            M=self.get_Matrix(self.boundry_points.X,'f')
            B=tf.reshape(self.boundry_points.F,[-1,1])
        if self.isBody: 
            dK=tf.reshape(self.body_points.dk,[-1,1])
            K=tf.reshape(self.body_points.k,[-1,1])
            q=tf.reshape(self.body_points.q,[-1,1])
            F=tf.reshape(self.body_points.F,[-1,1])
            M_=self.get_Matrix(self.body_points.X,'f')
            M_x=self.get_Matrix(self.body_points.X,'fx')
            M_xx=self.get_Matrix(self.body_points.X,'fxx')
            Md=dK*M_x+K*M_xx
            Bd=F-q
            if self.isBoundry:
                M=tf.concat((M,Md), axis=0)
                B=tf.concat((B,Bd), axis=0)
            else:
                M=Md
                B=Bd
            
        if not M==[]:
            C=tf_linsolve(M, B)
            self.C=tf.Variable(C)
            print(self.C)
            
    def get_Matrix(self,X,der='f'):
        X=tf.reshape(X,[-1,1])
        R=tf.reshape(self.r,[1,-1])
        Ic=tf.ones_like(X)        
        XX=X*self.Ir
        RR=Ic*R
        if der=='f':
            M=self.tf_F(XX,RR)
        elif der=='fx':
            M=self.tf_Fx(XX,RR)
        elif der=='fxx':
            M=self.tf_Fxx(XX,RR)
        return M 
    def np_predict(self,x,der='f'):
        shape=x.shape
        X=tf.convert_to_tensor(x,dtype=self.dtype)
        F=self.predict(X,der)
        return np.reshape(F.numpy(), shape) 
    
    def predict(self,x,der='f'):
        shape=x.shape
        M=self.get_Matrix(x,der)
        F=tf.linalg.matmul(M, self.C)
        return tf.reshape(F*self.par, shape) 
    