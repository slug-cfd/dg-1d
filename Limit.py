import numpy as np
import Interpolation as Interpolation

class Limit:
    def __init__(self, linedg):
        self.__linedg = linedg
        self.__params = linedg.params
        self.__lim = self.__params.LimitSolution()
        self.__nnodes = self.__params.nnodes()
        self.__nquads = self.__params.nquads()
        self.__nels   = self.__params.nels()
        self.__dx     = linedg.mesh.dx()
        self.ip = Interpolation.Interpolation(self.__nnodes, self.__nquads)

    def minmod(self,ubar):
        s = np.sum(np.sign(ubar))/ubar.size
        if (np.abs(s) == 1):
            m = s*np.amin(np.abs(ubar))
        else:
            m = 0
        return m

    def FindElementsToLimit(self):
        # Compute Cell averages
        self.ubar = np.zeros([self.__nels+2,1])
        for i in range(self.__nels+2):
            self.ubar[i] = np.sum( np.matmul( self.ip.W(),np.matmul(self.ip.B(),self.u[i,:]) ) )
        
        # compute interface fluxes
        ids = np.zeros([self.__nels,1])
        self.__v = np.zeros([self.__nels,2])
        for i in range(1,self.__nels+1):
            j = i-1
            self.__v[j,0] = self.ubar[i] - self.minmod( np.array([ self.ubar[i]-self.u[i-1,0], self.ubar[i] - self.ubar[i-1], self.ubar[i+1] - self.ubar[i]]) )
            self.__v[j,1] = self.ubar[i] + self.minmod( np.array([ self.ubar[i]-self.u[i-1,-1], self.ubar[i] - self.ubar[i-1], self.ubar[i+1] - self.ubar[i] ]) )

            
            if ( (np.abs(self.__v[j,0]-self.u[i,0]) > 1e-8) or (np.abs(self.__v[j,1]-self.u[i,-1]) > 1e-8) ):
                ids[j] = 1
        return ids

    def LimitSolution(self,u):
        self.ulimit = u
        self.u = np.zeros([self.__nels+2, self.__nnodes])
        self.u[1:self.__nels+1,:] = u

        if (self.__params.BoundaryConditions() == "periodic"):
            self.u[0,:] = u[-1,:]
            self.u[-1,:] = u[0,:]
        else:
            self.u[0,:] = np.ones([self.__nnodes])*self.__params.LeftBC()
            self.u[-1,:] = np.ones([self.__nnodes])*self.__params.RightBC()
        
        ids = self.FindElementsToLimit()
        ids[0] = 1
        self.__el_lim_ids = np.nonzero(ids > 0)
        if (self.__el_lim_ids[0].size != 0):
            if (self.__linedg.params.Limiter() == "pi1"):
                # print("PI1 Limiting")
                self.PI1()
            elif(self.__linedg.params.Limiter() == "muscl"):
                # print("MUSCL Limiting")
                self.MUSCL()

    
    #PI1 presented by Osher in "Convergence of generalized muscl schemes" 1985
    def PI1(self):
        self.__X = self.__linedg.mesh.X()
        self.dubar = np.zeros([self.__nels+2,1])
        for i in range(self.__nels+2):
            self.dubar[i] = np.sum( np.matmul( self.ip.W(), np.matmul(self.ip.D(), self.u[i,:])))/self.__dx
        
        for i in range(self.__el_lim_ids[0].shape[0]):
            # define element under consideration
            el = self.__el_lim_ids[0][i]
            elpGC = el+1 # element id plus the ghost cell
            xc = (self.__X[el,0] + 0.5*self.__dx) # element center
            
            r = np.array([self.dubar[elpGC], (self.ubar[elpGC+1] - self.ubar[elpGC])/(self.__dx/2), (self.ubar[elpGC] - self.ubar[elpGC-1])/(self.__dx/2) ])
            for j in range(self.__nnodes):
                self.ulimit[el,j] = self.ubar[elpGC] + (self.__X[el,j] - xc)*self.minmod(r)

    def MUSCL(self):
        self.__X = self.__linedg.mesh.X()
        self.dubar = np.zeros([self.__nels+2,1])
        for i in range(self.__nels+2):
            self.dubar[i] = np.sum( np.matmul( self.ip.W(), np.matmul(self.ip.D(), self.u[i,:])))/self.__dx
        
        for i in range(self.__el_lim_ids[0].shape[0]):
            # define element under consideration
            el = self.__el_lim_ids[0][i]
            elpGC = el+1 # element id plus the ghost cell
            xc = (self.__X[el,0] + 0.5*self.__dx) # element center
            
            r = np.array([self.dubar[elpGC], (self.ubar[elpGC+1] - self.ubar[elpGC])/(self.__dx), (self.ubar[elpGC] - self.ubar[elpGC-1])/(self.__dx) ])
            for j in range(self.__nnodes):
                self.ulimit[el,j] = self.ubar[elpGC] + (self.__X[el,j] - xc)*self.minmod(r)