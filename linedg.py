import numpy as np
import Interpolation as interpolation

class linedg:
    def __init__(self, mesh, params, equations):
        self.params = params
        self.__dim   = 1
        self.__neqs  = params.neqs()
        self.__order = params.order()
        self.__nnodes = self.__order+1
        self.__nels  = params.nels()
        self.__nquads = params.nquads()
        self.ip = interpolation.Interpolation(self.__nnodes, self.__nquads)
        self.mesh  = mesh
        self.mass = np.matmul(np.transpose(self.ip.B()), np.matmul(self.ip.W(), self.ip.B()) )
        self.invMass = np.linalg.inv(self.mass)
        self.__q = np.zeros([self.__nels,self.__nnodes,self.__neqs])
        self.equations = equations
        self.__leftBC = np.zeros([self.__neqs])
        self.__rightBC = np.zeros([self.__neqs])

    def order(self) -> int:
        return self.__order

    def q(self) -> np.array:
        return self.__q

    def Flux(self, u: np.array) -> np.array:
        F = self.equations.Flux(u)
        return F*self.mesh.J()*self.mesh.invJ()

    def SetBC(self):
        if (self.params.BoundaryConditions() == "periodic"):
            self.__leftBC = self.u[self.__nels-1, self.__nnodes-1,:]
            self.__rightBC = self.u[0,0,:]
        elif (self.params.BoundaryConditions() == "outflow"):
            self.__leftBC = self.params.LeftBC()
            self.__rightBC = self.params.RightBC()
        # elif (self.params.BoundaryConditions() == "outflow"):


    def AssembleElement(self, u: np.array) -> np.array:
        self.u = u
        self.SetBC()
        q = np.zeros([self.__nnodes,self.__neqs])
        U = np.zeros([self.__nels, self.__nquads, self.__neqs])
        fstar = np.zeros([2,self.__neqs])
        for iel in range(self.__nels):
            # Compute u at quadrature points
            for ieq in range(self.__neqs):
                U[iel,:,ieq] = np.matmul(self.ip.B(),self.u[iel,:,ieq])

            # Compute flux at quadrature points 
            F = np.zeros(U.shape)
            for iq in range(self.__nquads):
                # print(U[iel,iq,:])
                F[iel,iq,:] = self.Flux(U[iel,iq,:])


            for ieq in range(self.__neqs):
                #Compute volume term
                q[:,ieq] = -np.matmul( np.transpose(self.ip.D()), np.matmul(self.ip.W(), F[iel,:,ieq]) )
                # Compute numerical flux
                if (iel == 0):
                    leftBC = self.equations.Prim2Cons(self.__leftBC)
                    temp = self.RiemannSolver(leftBC,self.u[iel,0,:])
                    fstar[0,ieq] = temp[ieq] 
                    # fstar[0,ieq] = self.RiemannSolver(self.__leftBC[ieq],self.u[iel,0,ieq])

                    temp = self.RiemannSolver(self.u[iel,-1,:], self.u[iel+1,0,:])
                    fstar[-1,ieq] = temp[ieq] 
                    # fstar[-1,ieq] = self.RiemannSolver(self.u[iel,-1,ieq], self.u[iel+1,0,ieq])
                elif (iel == self.__nels-1):
                    temp = self.RiemannSolver(self.u[iel-1,-1,:], self.u[iel,0,:])
                    fstar[0,ieq] = temp[ieq] 

                    rightBC = self.equations.Prim2Cons(self.__rightBC)
                    temp =self.RiemannSolver(self.u[iel,-1,:], rightBC)
                    fstar[-1,ieq] = temp[ieq]
                else:
                    temp = self.RiemannSolver(self.u[iel-1,-1,:], self.u[iel,0,:])
                    fstar[0,ieq] = temp[ieq] 

                    temp = self.RiemannSolver(self.u[iel,-1,:], self.u[iel+1,0,:])
                    fstar[-1,ieq] = temp[ieq] 
                
                q[0,ieq] -= fstar[0,ieq]
                q[-1,ieq] += fstar[1,ieq]
                self.__q[iel,:,ieq] = np.matmul(self.invMass, q[:,ieq])
        return -self.__q/self.mesh.detJ()

    ######################################################
    #
    # Riemann Solver Functions
    #
    #######################################################

    def RiemannSolver(self, uleft: np.array, uright: np.array) -> np.array:
        rs = self.params.RiemannSolver()
        if (rs == "upwind"):
            c = np.max(self.u)
            fhat = self.Upwind(c,uleft,uright)
        elif (rs == "godunov"):
            fhat = self.Godunov(uleft,uright)
        elif (rs == "lf"):
            fhat = self.LF(uleft,uright)
        elif (rs == "llf"):
            fhat = self.LLF(uleft,uright)
        return fhat

    def Upwind(self, c, uleft: np.array, uright: np.array) -> np.array:
        if (c >= 0 ):
            ustar = uleft
        elif (c < 0):
            ustar = uright
        fstar = self.Flux(ustar)
        return fstar
    
    def Godunov(self, uleft: np.array, uright: np.array) -> np.array:
        s = 0.5*(uleft + uright)
        if (uleft >= uright):
            if (s >= 0):
                ustar = uleft
            else:
                ustar = uright
        elif (uleft < uright):
            if (uleft >= 0):
                ustar = uleft
            elif (uright <= 0):
                ustar = uright
            elif (uleft < 0 and uright > 0):
                ustar = 0
        Fstar = self.Flux(ustar)
        return Fstar
    
    def LF(self, uleft: np.array, uright: np.array) -> np.array:
        Cmax = np.amax(self.equations.Dflux(self.u))
        FL = self.Flux(uleft)
        FR = self.Flux(uright)
        avgState = 0.5*(FL + FR)
        jump     = uright - uleft
        Fstar    = avgState - 0.5*Cmax*jump
        return Fstar

    def LLF(self, uleft: np.array, uright: np.array) -> np.array:

        assert(self.equations.IsStatePhysical(uleft))
        assert(self.equations.IsStatePhysical(uright))


        Cmax = np.amax([self.equations.Dflux(uleft),self.equations.Dflux(uright)])
        FL = self.Flux(uleft)
        FR = self.Flux(uright)
        avgState = 0.5*(FL + FR)
        jump     = uright - uleft
        Fstar    = avgState - 0.5*Cmax*jump
        return Fstar 





    



    