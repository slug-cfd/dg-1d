import numpy as np
import Interpolation as ip


class Mesh:
    def __init__(self,domain,numOfEls,numOfNodes,numOfQuads):
        self.__xmin = domain[0]
        self.__xmax = domain[1]
        self.__K    = numOfEls
        self.__nN   = numOfNodes
        self.__nQ   = numOfQuads
        self.__ip   = ip.Interpolation(self.__nN, self.__nQ)
        self.__dx   = (self.__xmax-self.__xmin)/self.__K
        self.__J    = self.__dx
        self.__detJ = np.abs(self.__J)
        self.__invJ = 1/self.__J
        
    def Xmin(self):
        return self.__xmin
    def Xmax(self):
        return self.__xmax

    #number of elements    
    def nels(self):
        return self.__nels
    def dx(self):
        return self.__dx
    def X(self, nk, inode):
        return self.__dx*nk + self.__ip.Nodes()[inode]*self.__dx
    def X(self):
        self.__X = np.zeros([self.__K, self.__nN])
        for i in range(self.__K):
            self.__X[i,:] = (i*self.__dx + self.__xmin)+ self.__ip.xnodes()*self.__dx
        return self.__X

    def J(self):
        return self.__J
    def detJ(self):
        return self.__detJ
    def invJ(self):
        return self.__invJ
        

