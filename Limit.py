import numpy as np
import Interpolation as Interpolation

class LimitMOOD:
    def __init__(self,linedg):
        self.__linedg = linedg
        self.__params = linedg.params
        self.__nnodes = self.__params.nnodes()
        self.__nquads = self.__params.nquads()
        self.__nels   = self.__params.nels()
        self.__neqs   = self.__params.neqs()
        self.__dx     = linedg.mesh.dx()
        self.ip = Interpolation.Interpolation(self.__nnodes, self.__nquads)
        self.ubar    = np.zeros([self.__nels+2,self.__neqs])
        self.pbar    = np.zeros([self.__nels+2])
        self.umodal  = np.zeros([self.__nels, self.__nnodes, self.__neqs])
        
        # local flow compressibility parameter
        self.sigma_v = 5.0

        # pressure gradient check
        self.sigma_p = 5.0

        pass

    def LimitSolution(self, u: np.array) -> np.array:
        self.ulimit = u
        self.ughost = np.zeros([self.__nels+2,self.__nnodes, self.__neqs])
        self.ughost[1:self.__nels+1,:,:] = u

        leftBC = self.__linedg.equations.Prim2Cons(self.__params.LeftBC()) 
        rightBC = self.__linedg.equations.Prim2Cons(self.__params.RightBC()) 
        for i in range(self.__nnodes):
            self.ughost[0,i,:] = leftBC
            self.ughost[-1,i,:] = rightBC

        # Compute Average
        for i in range(self.__nels+2):
            self.ubar[i,:] = self.ComputeElementAverage(i,self.ughost)
            self.pbar[i] = self.__linedg.equations.Pressure(self.ubar[i,:])

        mood_finish = False
        mode = np.full([self.__nels+2], self.__nnodes-1, dtype=int) 
        while not mood_finish:
            self.Truncate = self.FindElementsToLimit() 
            el_trunc_list = np.nonzero(self.Truncate > 0)
            print(el_trunc_list[0].size)
            if el_trunc_list[0].size == 0:
                mood_finish = True
            # TODO (mjrodriguez): Moved this loop within an else loop and iterate through only the truncation list 
            for iel in range(1,self.__nels+1):
                if self.Truncate[iel] and mode[iel] > 0:
                    mode[iel] = self.TruncateModalSolution(iel, mode[iel])
            
        
        print("Truncate List")
        print(self.Truncate)


        return self.ulimit 

    def FindElementsToLimit(self) -> np.array:
        truncate = np.full((self.__nels+2), False, dtype=bool)
        for i in range(1,self.__nels+1):

            dmp_check = True
            dens = self.ubar[i,0]
            # Checking PAD and CAD conditions
            if np.isnan(self.pbar[i]) or np.isnan(dens):
                truncate[i] = True
                dmp_check = False
            if np.isinf(self.pbar[i]) or np.isinf(dens):
                truncate[i] = True
                dmp_check = False
            if self.pbar[i] < 0 or dens < 0:
                truncate[i] = True
                dmp_check = False

            if dmp_check:
                # Compressibility and strogn shocks check
                if self.StrongCompressibilityCheck(i) or self.StrongPressureCheck(i):
                    # print("Strong compressible pressure grad present, iel = ", i)

                    minrho, maxrho = self.LocalMinMaxRho(i)

                    # new extrema in plateau check
                    if maxrho - minrho >= self.__dx**3:
                        # print("New extrema in plateau")
                        # DMP
                        if  self.ubar[i,0] < minrho or self.ubar[i,0] > maxrho:
                            # print("outside DMP")
                            truncate[i] = True
                            # Check if smooth extrema
                            Cmin, Cmax = self.ComputeU2MinMax(i)
                            delta = self.__dx
                            if Cmin*Cmax > -delta and ( np.max( np.array([Cmin,Cmax]) ) < delta or  np.abs(Cmin/Cmax) >= 0.5 ):
                                # If in smooth extrema then solution in element regains admissibility
                                truncate[i] = False
                                # print("iel = ", i, " Recovers Admissibility")

        return truncate


    def StrongCompressibilityCheck(self, iel: int) -> bool:
        uprim_r = self.__linedg.equations.Cons2Prim(self.ubar[iel+1,:])
        uprim_l = self.__linedg.equations.Cons2Prim(self.ubar[iel-1,:])
        div_v = (uprim_r[1] - uprim_l[1])/(2*self.__dx)
        if div_v < -self.sigma_v:
            return True

        return False

    def StrongPressureCheck(self, iel: int) -> bool:
        pmin = np.min(np.array([ self.pbar[iel+1], self.pbar[iel-1] ]))
        gradp = np.abs(self.pbar[iel+1] - self.pbar[iel-1]) / (2*self.__dx*pmin)

        if gradp > self.sigma_p:
            return True

        return False

    # Compute the min and max density of the neighborhood iel-1, iel, iel+1 where iel is the element under consideration
    def LocalMinMaxRho(self, iel: int) -> tuple:
        maxrho = np.max(np.array([ self.ubar[iel-1,0], self.ubar[iel, 0], self.ubar[iel+1,0]  ]))
        minrho = np.min(np.array([ self.ubar[iel-1,0], self.ubar[iel, 0], self.ubar[iel+1,0]  ]))

        return minrho, maxrho

    def ComputeU2MinMax(self, iel: int) -> tuple:
        D2 = np.zeros(3)
        # Currently we are only considereing element iel but we need to compute second order derivatives at adjacent elements, i.e. iel-1, iel, and iel+1.
        # So we iterate through -1, 0, 1 to shift the element index. It's sliding window-like. 
        # ielx is then the cetner element and we can compute centered finite difference second order derivatives for those elements.
        for ix in range(-1,2):
            ielx = iel-ix
            D2[ix+1] = ( self.ubar[ielx-1] - 2*self.ubar[ielx] + self.ubar[ielx+1] ) / self.__dx**2
        
        Cmin = np.min(D2)
        Cmax = np.max(D2)

        return Cmin, Cmax


    def TruncateModalSolution(self,iel: int, im: int) -> int:
        self.ComputeElementModalSolution(iel)
        self.umodal[iel,im,:] = 0.0
        for ieq in range(self.__neqs):
            self.ulimit[iel,:,ieq] = self.ip.V@self.umodal[iel,:,ieq]
        self.ubar[iel,:] = self.ComputeElementAverage(iel, self.ulimit)
        self.pbar = self.__linedg.equations.Pressure(self.ubar[iel,:]) 

        return im-1



    def ComputeElementAverage(self, iel: int, u: np.array) -> np.array:
        ubar = np.zeros([self.__neqs])
        for ieq in range(self.__neqs):
            ubar[ieq] = np.sum( np.matmul( self.ip.W(),np.matmul(self.ip.B(), u[iel,:,ieq]) ) )
        return ubar 

    def ComputeElementModalSolution(self,iel: int):
        for ieq in range(self.__neqs):
            self.umodal[iel,:,ieq] = self.ip.Vinv@self.ulimit[iel,:,ieq]
        pass

class Limit:
    def __init__(self, linedg):
        self.__linedg = linedg
        self.__params = linedg.params
        self.__lim = self.__params.LimitSolution()
        self.__nnodes = self.__params.nnodes()
        self.__nquads = self.__params.nquads()
        self.__nels   = self.__params.nels()
        self.__neqs   = self.__params.neqs()
        self.__dx     = linedg.mesh.dx()
        self.ip = Interpolation.Interpolation(self.__nnodes, self.__nquads)

    def minmod(self,ubar):
        # print(ubar)
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

    def LimitSolution(self, u: np.array, ieq: int):

        self.ulimit = u
        self.u = np.zeros([self.__nels+2, self.__nnodes])
        # print(u.shape)
        self.u[1:self.__nels+1,:] = u

        if (self.__params.BoundaryConditions() == "periodic"):
            self.u[0,:] = u[-1,:]
            self.u[-1,:] = u[0,:]
        else:
            self.u[0,:] = np.ones([self.__nnodes])*self.__linedg.equations.Prim2Cons(self.__params.LeftBC())[ieq]
            self.u[-1,:] = np.ones([self.__nnodes])*self.__linedg.equations.Prim2Cons(self.__params.RightBC())[ieq]
        
        self.ids = self.FindElementsToLimit()
        # self.ids[0] = 1
        self.el_lim_ids = np.nonzero(self.ids > 0)
        if (self.el_lim_ids[0].size != 0):
            if (self.__linedg.params.Limiter() == "pi1"):
                # print("PI1 Limiting")
                self.PI1()
            elif(self.__linedg.params.Limiter() == "muscl"):
                # print("MUSCL Limiting")
                self.MUSCL()
        pass
    
    #PI1 presented by Osher in "Convergence of generalized muscl schemes" 1985
    def PI1(self):
        self.__X = self.__linedg.mesh.X()
        self.dubar = np.zeros([self.__nels+2,1])
        for i in range(self.__nels+2):
            self.dubar[i] = np.sum( np.matmul( self.ip.W(), np.matmul(self.ip.D(), self.u[i,:])))/self.__dx
        
        for i in range(self.el_lim_ids[0].shape[0]):
            # define element under consideration
            el = self.el_lim_ids[0][i]
            elpGC = el+1 # element id plus the ghost cell
            xc = (self.__X[el,0] + 0.5*self.__dx) # element center
            
            r = np.array([self.dubar[elpGC], (self.ubar[elpGC+1] - self.ubar[elpGC])/(self.__dx/2), (self.ubar[elpGC] - self.ubar[elpGC-1])/(self.__dx/2) ])
            for j in range(self.__nnodes):
                self.ulimit[el,j] = self.ubar[elpGC] + (self.__X[el,j] - xc)*self.minmod(r)
        
        pass

    def MUSCL(self):
        self.__X = self.__linedg.mesh.X()
        self.dubar = np.zeros([self.__nels+2,1])
        for i in range(self.__nels+2):
            self.dubar[i] = np.sum( np.matmul( self.ip.W(), np.matmul(self.ip.D(), self.u[i,:])))/self.__dx
        
        for i in range(self.el_lim_ids[0].shape[0]):
            # define element under consideration
            el = self.el_lim_ids[0][i]
            elpGC = el+1 # element id plus the ghost cell
            xc = (self.__X[el,0] + 0.5*self.__dx) # element center
            
            r = np.array([self.dubar[elpGC], (self.ubar[elpGC+1] - self.ubar[elpGC])/(self.__dx), (self.ubar[elpGC] - self.ubar[elpGC-1])/(self.__dx) ])
            for j in range(self.__nnodes):
                self.ulimit[el,j] = self.ubar[elpGC] + (self.__X[el,j] - xc)*self.minmod(r)