import numpy as np
import SimulationParameters as simp
import Interpolation as Interpolation
import Mesh as m

def Shock(params, x):
    ip = Interpolation.Interpolation(params.nnodes(), params.nquads())
    uinit = np.zeros([params.nels(),params.nnodes()])
    for i in range(params.nels()):
        #compute center of cell
        xc = np.sum( np.matmul(ip.W(),np.matmul(ip.B(),x[i,:])))
        # print(xc)
        if (xc < params.ShockLoc()):
            uinit[i,:] = params.LeftBC()
        elif (xc >= params.ShockLoc()):
            uinit[i,:] = params.RightBC()
    return uinit

def Rarefaction(params, x):
    ip = Interpolation.Interpolation(params.nnodes(), params.nquads())
    uinit = np.zeros([params.nels(),params.nnodes()])
    for i in range(params.nels()):
        #compute center of cell
        xc = np.sum( np.matmul(ip.W(),np.matmul(ip.B(),x[i,:])))
        # print(xc)
        if (xc < params.ShockLoc()):
            uinit[i,:] = params.LeftBC()
        elif (xc >= params.ShockLoc()):
            uinit[i,:] = params.RightBC()
        
    return uinit

def ShockRarefaction(params,x):
    ip = Interpolation.Interpolation(params.nnodes(), params.nquads())
    uinit = np.zeros([params.nels(),params.nnodes()])
    for i in range(params.nels()):
        xc = np.sum( np.matmul(ip.W(),np.matmul(ip.B(),x[i,:])))
        if (xc <= 0.3):
            uinit[i,:] = 2
        elif (xc > 0.3 and xc <= 0.6):
            uinit[i,:] = -1
        elif (xc > 0.6):
            uinit[i,:] = 3
    return uinit

def DoubleShock(params,x):
    ip = Interpolation.Interpolation(params.nnodes(), params.nquads())
    uinit = np.zeros([params.nels(),params.nnodes()])
    for i in range(params.nels()):
        xc = np.sum( np.matmul(ip.W(),np.matmul(ip.B(),x[i,:])))
        if (xc <= 0.3):
            uinit[i,:] = 4
        elif (xc > 0.3 and xc <= 0.6):
            uinit[i,:] = 2
        elif (xc > 0.6):
            uinit[i,:] = -1
    return uinit 


    # # for j in range(Par.p+1):
    # #     if (x[i,j] > shockLoc):
    # #         uinit[i,j] = ur
    # #     else:
    # #         uinit[i,j] = ul
    # return uinit

class problem:
    def __init__ (self, problem):
        self.__problem = problem

    def GetProblem(self):
        return self.__problem

    def SetProblem(self):
        if (self.__problem == 1):
            neqs = 1
            order = 4
            nquads = 2*order
            nels = 100
            domain = np.array([-1,1])
            maxtime = 0.3
            cfl = 0.2
            leftBC = 2.0
            rightBC = 1.0
            shockLoc = -0.5
            plotSol=True

            params = simp.Parameters(neqs, order, nquads, nels, domain, cfl, maxtime, "rk4", "outflow", "godunov", True, "pi1", leftBC, rightBC, shockLoc)
            mesh = m.Mesh(params.domain(), params.nels(), params.nnodes(), params.nquads())
            u = Shock(params, mesh.X())
        elif(self.__problem == 2):
            neqs = 1
            order = 1
            nquads = 2*order
            nels = 100
            domain = np.array([0,1])
            maxtime = 0.2
            cfl = 0.2
            leftBC = -1.0
            rightBC = 1.0
            shockLoc = 0.5
            plotSol=True

            params = simp.Parameters(neqs, order, nquads, nels, domain, cfl, maxtime, "rk4", "outflow", "godunov", True, "pi1", leftBC, rightBC, shockLoc)
            mesh = m.Mesh(params.domain(), params.nels(), params.nnodes(), params.nquads())
            u = Rarefaction(params, mesh.X())
        elif(self.__problem == 3):
            neqs = 1
            order = 4
            nquads = 2*order
            nels = 100
            domain = np.array([0,1])
            maxtime = 0.3
            cfl = 0.2
            leftBC = 4.0
            rightBC = -1.0
            shockLoc = -0.5
            plotSol=True

            params = simp.Parameters(neqs, order, nquads, nels, domain, cfl, maxtime, "rk4", "outflow", "godunov", True, "pi1", leftBC, rightBC, shockLoc)
            mesh = m.Mesh(params.domain(), params.nels(), params.nnodes(), params.nquads())
            u = DoubleShock(params, mesh.X()) 
        elif(self.__problem == 4):
            neqs = 1
            order = 4
            nquads = 2*order
            nels = 100
            domain = np.array([0,1])
            maxtime = 0.3
            cfl = 0.2
            leftBC = 2.0
            rightBC = 3.0
            shockLoc = -0.5
            plotSol=True

            params = simp.Parameters(neqs, order, nquads, nels, domain, cfl, maxtime, "rk4", "outflow", "godunov", True, "pi1", leftBC, rightBC, shockLoc)
            mesh = m.Mesh(params.domain(), params.nels(), params.nnodes(), params.nquads())
            u = ShockRarefaction(params, mesh.X())
        elif (self.__problem == 5):
            neqs = 1
            order = 5
            nquads = 2*order
            nels = 30
            domain = np.array([-1,1])
            maxtime = 0.4
            cfl = 0.8
            leftBC = 2.0
            rightBC = 1.0
            shockLoc = -0.5
            plotSol=True

            params = simp.Parameters(neqs, order, nquads, nels, domain, cfl, maxtime, "rk4", "periodic", "lf", True, "pi1", leftBC, rightBC, shockLoc, plotSol)
            mesh = m.Mesh(params.domain(), params.nels(), params.nnodes(), params.nquads())
            u = 0.5 + np.sin(np.pi*mesh.X())
        elif (self.__problem == 6):
            neqs = 1
            order = 5
            nquads = 2*order
            nels = 30
            domain = np.array([-3,3])
            maxtime = 0.8
            cfl = 0.8
            leftBC = 2.0
            rightBC = 1.0
            shockLoc = -0.5
            plotSol=True

            params = simp.Parameters(neqs, order, nquads, nels, domain, cfl, maxtime, "rk4", "periodic", "lf", True, "pi1", leftBC, rightBC, shockLoc, plotSol)
            mesh = m.Mesh(params.domain(), params.nels(), params.nnodes(), params.nquads())
            sigma = 0.45
            mu = -0.5
            u = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*( (mesh.X() - mu)/(sigma) )**2 )

        return params, mesh, u
