import numpy as np
import Interpolation as interpolation

class linedg:
    def __init__(self, mesh, params):
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
        self.__q = np.zeros([self.__nels,self.__nnodes])
    def order(self):
        return self.__order

    def q(self):
        return self.__q

    def Flux(self,u):
        # Compute flux
        F = np.multiply(u,u)
        # F = u
        return F*self.mesh.J()*self.mesh.invJ()

    def SetBC(self):
        if (self.params.BoundaryConditions() == "periodic"):
            self.__leftBC = self.u[self.__nels-1, self.__nnodes-1]
            self.__rightBC = self.u[0,0]
        elif (self.params.BoundaryConditions() == "outflow"):
            self.__leftBC = self.params.LeftBC()
            self.__rightBC = self.params.RightBC()
        # elif (self.params.BoundaryConditions() == "outflow"):


    def AssembleElement(self,u):
        self.u = u
        self.SetBC()
        q = np.zeros([self.__nnodes,1])
        fstar = np.zeros([2,1])
        for iel in range(self.__nels):
            # u at quadrature points
            U = np.matmul( self.ip.B(), self.u[iel,:])
            # Compute reference flux
            F = self.Flux(U)
            #Compute volume term
            q = -np.matmul( np.transpose(self.ip.D()), np.matmul(self.ip.W(), F) )
            if (iel == 0):
                fstar[0] = self.RiemannSolver(self.__leftBC,self.u[iel,0])
                fstar[-1] = self.RiemannSolver(self.u[iel,-1], self.u[iel+1,0])
            elif (iel == self.__nels-1):
                fstar[0] = self.RiemannSolver(self.u[iel-1,-1], self.u[iel,0])
                fstar[-1] =self.RiemannSolver(self.u[iel,-1], self.__rightBC)
            else:
                fstar[0] = self.RiemannSolver(self.u[iel-1,-1], self.u[iel,0])
                fstar[-1] = self.RiemannSolver(self.u[iel,-1], self.u[iel+1,0])
            
            q[0] -= fstar[0]
            q[-1] += fstar[1]
            self.__q[iel,:] = np.matmul(self.invMass, q)
        return -self.__q/self.mesh.detJ()

    ######################################################
    #
    # Riemann Solver Functions
    #
    #######################################################

    def RiemannSolver(self, uleft, uright):
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

    def Upwind(self, c, uleft, uright):
        if (c >= 0 ):
            ustar = uleft
        elif (c < 0):
            ustar = uright
        fstar = self.Flux(ustar)
        return fstar
    
    def Godunov(self, uleft, uright):
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
    
    def LF(self, uleft, uright):
        Cmax = np.amax(self.u)
        FL = self.Flux(uleft)
        FR = self.Flux(uright)
        avgState = 0.5*(FL + FR)
        jump     = uright - uleft
        Fstar    = avgState - 0.5*Cmax*jump
        return Fstar

    def LLF(self, uleft, uright):
        Cmax = np.amax([uleft,uright])
        FL = self.Flux(uleft)
        FR = self.Flux(uright)
        avgState = 0.5*(FL + FR)
        jump     = uright - uleft
        Fstar    = avgState - 0.5*Cmax*jump
        return Fstar 


        




    



    