from scipy.spatial import distance
import numpy as np
from kmeans import kmeans

    
st="/home/misero/machine/3D_spatial_network.txt"
X=np.loadtxt(st,delimiter=',')
X=X[:,1:]
path="/home/misero/machine"
text_file = open(path+"/kmppRo_python.csv", "w")
text_file2 = open(path+"/kmppRo_tiempòs.csv", "w")
for i in range(15):
    km=kmeans(X,15,init ="kmpp",n_cores=8)
    km.fit()
    t=""
    for i in km.getCentroides():
        for j in i:
            t+=np.array2string(j)+","        
        t+="\r\n"    
    km=None
    text_file.write(t)
    text_file2.write(km.getSTiempos())
text_file.close()
text_file2.close()


