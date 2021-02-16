import numpy as np

class Parameters:
    def __init__(self, neqs=1, order=2, quads=3*2+1, numOfEls=30, domain=np.array([0,1]), courantNumber=0.8, maxTime=0.25, timeIntegration="rk4" ,boundaryCondition="periodic", riemannSolver="godunov", limitBool=False, limiter="pi1",leftBC=0, rightBC=0, shockLoc=0, plotSol=True):
        self.__order = order
        self.__neqs  = neqs
        self.__p     = order+1
        self.__quads = quads
        self.__nK    = numOfEls
        self.__domain = domain
        self.__courantNumber     = courantNumber
        self.__maxTime = maxTime
        self.__timeIntegration = timeIntegration
        self.__boundaryCondition = boundaryCondition
        self.__limit  = limitBool
        self.__limiter = limiter
        self.__riemannSolver = riemannSolver
        self.__shockLoc = shockLoc
        self.__leftBC = leftBC
        self.__rightBC = rightBC
        self.__plotSol = plotSol
        
    def neqs(self):
        return self.__neqs
    def order(self):
        return self.__order
    def nnodes(self):
        return self.__p
    def nquads(self):
        return self.__quads
    def nels(self):
        return self.__nK
    def domain(self):
        return self.__domain
    def LeftBC(self):
        return self.__leftBC
    def RightBC(self):
        return self.__rightBC
    def ShockLoc(self):
        return self.__shockLoc
    def CourantNumber(self):
        return self.__courantNumber
    def TimeIntegration(self):
        return self.__timeIntegration
    def MaxTime(self):
        return self.__maxTime
    def BoundaryConditions(self):
        return self.__boundaryCondition
    def LimitSolution(self):
        return self.__limit
    def Limiter(self):
        return self.__limiter
    def RiemannSolver(self):
        return self.__riemannSolver
    def PlotSol(self):
        return self.__plotSol
    def UpdatePlotSol(self, plotCond):
        self.__plotSol = plotCond