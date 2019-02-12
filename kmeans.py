import numpy as np
from random import sample
from scipy.spatial import distance
import time
from multiprocessing import Pool


class kmeans:
    def __init__(self, X, k=3, init="random",n_cores=4):
        self.k=k
        self.X=X
        self.centroides=None
        self.inicio=False
        self.init=init        
        self.n_cores=n_cores
            
    def fit(self):  
        tinit=""
        if self.init=="random":
            start=time.time()
            self.centroides=self.c_aleatorios()
            end=time.time()
            tinit=(end-start)
        elif self.init =="kmpp":            
            start=time.time()
            self.centroides=self.c_kmeanspp()
            end=time.time()
            tinit=(end-start)
        mdist=np.zeros((self.X.shape[0]),dtype=int)
        start=time.time()
        xt=0
        
        while not self.converge():  
            print("..",xt)
            self.inicio=True   
            mdist=None
            clusters=None
            cant=None
            #Distancias entre centroides y puntos                        
            pool=Pool(self.n_cores)
            mdist=np.asarray(pool.map(self.distancias, self.X))            
            pool.close()
            pool.join()

            clusters=np.zeros((self.k, self.X[0].shape[0]))
            cant=np.zeros((self.k,1))  
            for x in range(mdist.shape[0]):
                for c in range(self.k):                    
                    if mdist[x] == c:                        
                        for i in range(self.X[0].shape[0]):                                                    
                            clusters[c][i]=clusters[c][i]+self.X[x][i]                        
                        cant[c]=cant[c]+1
            for x in range(self.k):                    
                    for i in range(self.X[0].shape[0]):
                        clusters[x][i]=clusters[x][i]/cant[x]                           
            self.old_centroides=None
            self.old_centroides=self.centroides    
            self.centroides=None                         
            self.centroides=clusters       
            xt=xt+1                                       
        end=time.time() 
        self.clusters=mdist           
        self.tiempos=str((end-start))    
        print(self.k,",",((end-start)+tinit),",",(end-start),",",tinit,"/r/n")        

    def distancias(self, vector):
        dist_m=0
        for c in range(len(self.centroides)):
            dist=distance.euclidean(vector,self.centroides[c])
            if c==0:                    
                dist_m= c
                old_dist=dist                    
            elif  old_dist > dist:
                dist_m= c
                old_dist=dist 
        return dist_m

    def getSTiempos(self):
        return self.tiempos

    def getCentroides(self):
        return self.centroides

    def getClusters(self):
        cl=[]        
        for x in range(self.k):              
            it=[]          
            for y in range(self.X.shape[0]):                
                if self.clusters[y] == x:
                    it.append(self.X[y])
            cl.append(it)       
        return cl

    def converge(self):        
        if not self.inicio:
            return False
        elif np.unique(self.centroides,axis=0).shape[0]!=self.centroides.shape[0]:
            return False
        else:
            return np.array_equal(self.old_centroides,self.centroides)     
            
    def c_aleatorios(self):
        centroides = sample(list(self.X),self.k)
        return np.array(centroides)
    #Falta paralelizar el inicio.
    def c_kmeanspp(self):
        start=time.time()
        centroides=np.zeros((self.k,self.X[0].shape[0]))
        ri=np.random.randint(0, high=self.X.shape[0])
        centroides[0] = self.X[ri]
        distancias_cuadradas=np.zeros((self.X.shape[0],1))
        current=1        
        for i in range(1,self.k):
            for j in range(self.X.shape[0]):
                templ=[]  
                for l in range(current):
                    templ.append(distance.sqeuclidean(self.X[j],centroides[l]))                    
                distancias_cuadradas[j]=np.min(templ)                
            total=np.sum(distancias_cuadradas)            
            distancias_cuadradas=distancias_cuadradas/total
            distancias_cuadradas=np.cumsum(distancias_cuadradas)
            ran=np.random.rand()
            for m in range(distancias_cuadradas.shape[0]):
                if ran<distancias_cuadradas[m]:
                    centroides[i]=self.X[m]
                    current+=1
                    break   
        end=time.time()  
        #print("inicializacion  kmeans++ en :",(end-start)," s") 
        return centroides   




