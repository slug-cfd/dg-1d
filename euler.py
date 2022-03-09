import numpy as np


class Euler:
    def __init__(self,gamma=1.4):
        self.__gamma = gamma 

    def Gamma(self) -> float:
        return self.__gamma
    
    def Pressure(self,u: np.array) -> float:
        rhou_sqrd = np.multiply(u[1],u[1])  
        P = (self.Gamma()-1)*( u[2] - 0.5*( rhou_sqrd/u[0] ))
        # P = (self.Gamma()-1)*( u[2] - 0.5*np.divide( rhou_sqrd, u[0] ))
        return P

    # TODO (mrodrig6): implement transformation from conservative to primitive variables
    def Cons2Prim(self, consU):
        return 0

    def Prim2Cons(self, primU):
        consU = np.zeros(primU.shape)
        consU[0] = primU[0]
        consU[1] = np.multiply(primU[0],primU[2])
        consU[2] = primU[1]/(self.Gamma()-1) + 0.5*np.multiply(primU[0], np.multiply(primU[2],primU[2]))
        return consU

    def Flux(self,u: np.array) -> np.array:
        rhou_sqrd = u[1]*u[1]  
        pres = self.Pressure(u)
        F = np.zeros(u.shape)
        F[0] = u[1]
        # F[1] = np.divide(rhou_sqrd, u[0]) + pres
        F[1] = rhou_sqrd/u[0] + pres
        F[2] = (u[2] + pres)*u[1]
        F[2] = F[2]/u[0]
        return F
    
    def Dflux(self, u: np.array) -> float:
        # print("Dflux function:")
        # print(u)
        den = u[0]
        pres = self.Pressure(u)
        print(self.Gamma(), pres, den, self.Gamma()*pres/den)
        sound = np.sqrt(self.Gamma()*pres / den)
        vel = np.sqrt(u[1]*u[1]/den/den)
        print("vel = ", vel, " sound = ", sound)
        return vel + sound
    
    def IsStatePhysical(self,u: np.array)-> bool:

        den  = u[0]
        rhou = u[1]
        ener = u[2]

        if den < 0:
            print("Negative Density!!!")
            print("Current State:")
            print(u)
            return False
        if ener < 0:
            print("Negative Energy!!!")
            print("Current State:")
            print(u)
            return False
        
        pres = self.Pressure(u)

        if pres <= 0:
            print("Negative pressure!!!")
            print("Pressure = ", pres)
            print("Current State:")
            print(u)
            return False
        
        return True