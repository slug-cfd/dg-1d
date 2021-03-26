import numpy as np


class Euler:
    def __init__(self,gamma=1.4):
        self.__gamma = gamma 

    def Gamma(self):
        return self.__gamma
    
    def Pressure(self,u):
        rhou_sqrd = np.multiply(u[:,:,1],u[:,:,1])  
        P = (self.Gamma()-1)*( u[:,:,2] - 0.5*np.divide( rhou_sqrd, u[:,:,0] ))
        return P

    def Cons2Prim(self, consU):
        return 0

    def Prim2Cons(self, primU):
        consU = np.zeros(primU.shape)
        consU[:,:,0] = primU[:,:,0]
        consU[:,:,1] = np.multiply(primU[:,:,0],primU[:,:,2])
        consU[:,:,2] = primU[:,:,1]/(self.Gamma()-1) + 0.5*np.multiply(primU[:,:,0], np.multiply(primU[:,:,2],primU[:,:,2]))

        return consU

    def Flux(self,u):
        rhou_sqrd = np.multiply(u[:,:,1],u[:,:,1])  
        pres = self.Pressure(u)
        F = np.zeros(u.shape)
        F[:,:,0] = u[:,:,1]
        F[:,:,1] = np.divide(rhou_sqrd, u[:,:,0]) + pres
        F[:,:,2] = np.multiply(u[:,:,2] + pres, u[:,:,1])
        F[:,:,2] = np.divide(F[:,:,2], u[:,:,0])
        return F