import numpy as np
import linedg as linedg
import Limit as Limit

class time_integration:
    def __init__(self, linedg):
        self.__linedg = linedg
        self.__CourantNumber = linedg.params.CourantNumber()
        self.__integrator = linedg.params.TimeIntegration()
        self.__limit = Limit.Limit(linedg)

    def PrintLimitedCells(self):
        print(self.__limit.el_lim_ids)
        pass

    def Evolve(self, dt, t, uold):
        if (self.__integrator == "rk4"):
            unew = self.RK4(dt,t,uold)
        elif (self.__integrator == "ssprk3"):
            unew = self.RK4(dt,t,uold)
        return unew

    def RK4(self,dt,t,uold):
        k1 = np.zeros(uold.shape)
        k2 = np.zeros(uold.shape)
        k3 = np.zeros(uold.shape)
        k4 = np.zeros(uold.shape)
        
        k1 = dt*self.__linedg.AssembleElement(uold)
        if (self.__linedg.params.LimitSolution() == True):
            for ieq in range(self.__linedg.params.neqs()):
                self.__limit.LimitSolution(k1[:,:,ieq], ieq)

        k2 = dt*self.__linedg.AssembleElement(uold + 0.5*k1)
        if (self.__linedg.params.LimitSolution() == True):
            for ieq in range(self.__linedg.params.neqs()):
                self.__limit.LimitSolution(k2[:,:,ieq], ieq)

        k3 = dt*self.__linedg.AssembleElement(uold + 0.5*k2)
        if (self.__linedg.params.LimitSolution() == True):
            for ieq in range(self.__linedg.params.neqs()):
                self.__limit.LimitSolution(k3[:,:,ieq], ieq)

        k4 = dt*self.__linedg.AssembleElement(uold + k3)
        if (self.__linedg.params.LimitSolution() == True):
            for ieq in range(self.__linedg.params.neqs()):
                self.__limit.LimitSolution(k4[:,:,ieq], ieq)

        unew = uold + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        if (self.__linedg.params.LimitSolution() == True):
            for ieq in range(self.__linedg.params.neqs()):
                self.__limit.LimitSolution(unew[:,:,ieq], ieq)

        return unew
    
    def ssprk3(self, dt, t, uold):
        k1 = np.zeros(uold.shape)
        k2 = np.zeros(uold.shape)
        k3 = np.zeros(uold.shape)
        k2hat = np.zeros(uold.shape)

        k1 = uold + dt*self.__linedg.AssembleElement(uold)
        if (self.__linedg.params.LimitSolution() == True):
            for ieq in range(self.__linedg.params.neqs()):
                self.__limit.LimitSolution(k1[:,:,ieq], ieq)

        k2 = k1 + dt*self.__linedg.AssembleElement(k1)
        if (self.__linedg.params.LimitSolution() == True):
            for ieq in range(self.__linedg.params.neqs()):
                self.__limit.LimitSolution(k2[:,:,ieq], ieq)

        k2hat = (3.0*uold + k2)/4.0
        k3 = k2hat + dt*self.__linedg.AssembleElement(k2hat)
        if (self.__linedg.params.LimitSolution() == True):
            for ieq in range(self.__linedg.params.neqs()):
                self.__limit.LimitSolution(k3[:,:,ieq], ieq)

        unew = (uold + 2.0*k3)/3.0
        if (self.__linedg.params.LimitSolution() == True):
            for ieq in range(self.__linedg.params.neqs()):
                self.__limit.LimitSolution(unew[:,:,ieq], ieq)

        return unew

    def ComputeDt(self,u):
        umax = np.amax(u)
        # print("umax = ", umax, " order = ", self.__linedg.order()+1.0)
        # return self.__CourantNumber*self.__linedg.mesh.dx()/umax/((self.__linedg.order()+1.0)**2)
        return self.__CourantNumber*self.__linedg.mesh.dx()/umax/((self.__linedg.order()+1.0)**2.0)
