import numpy as np
import matplotlib.pyplot as plt
import Interpolation as Interpolation
import SimulationParameters as sp
import problems as pr
import Mesh as m
import time_int as tint
import linedg
import Limit
# from scipyimport sparse
from mpl_toolkits.mplot3d import Axes3D
import sys
np.set_printoptions(linewidth=250)



def PlotSolution(t, x, u, fig):
    xplot = x.ravel()
    fig.clf()
    colors = ['b-', 'r-', 'g-']
    if len(u.shape) == 3:
        n = u.shape[2]
        for i in range(n):
            uplot = u[:,:,i].ravel()
            plt.plot(xplot,uplot,colors[i])
    else:
        uplot = u.ravel()
        plt.plot(xplot,uplot,colors[0])

    
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Time = %1.3f" %t)
    plt.grid()
    plt.pause(0.05)
    plt.draw()

def Cons2Prim(consU):
    return primU

def Prim2Cons(primU):
    return consU

if __name__ == "__main__":



    
    probs = pr.EulerProblem(1)
    params, mesh, u = probs.SetProblem()
    fig = plt.figure()
    PlotSolution(0.0,mesh.X(), u[:,:,0], fig)
    # print("=================================================================")
    # print("*                    Simulation                                 *")
    # print("*                     DG Code                                   *")
    # print("*             Order = ", params.order() )
    # print("*         Number of Elements = ", params.nels())
    # print("*            CFL Constant = ", params.CourantNumber())
    # print("*              Problem  = ", probs.GetProblem())
    # print("=================================================================")

    # # params.UpdatePlotSol(False)
    # utest = np.ones([params.nels(), params.nnodes()])



    # ldg = linedg.linedg(mesh, params)
    # ti = tint.time_integration(ldg)
    # # lim = Limit.Limit(ldg)
    # # ubar = lim.LimitSolution(u)

    # t = 0.0
    # nsteps = 0
    # fig = plt.figure()
    # while t < params.MaxTime():
    #     dt = ti.ComputeDt(u)
    #     if (dt > params.MaxTime()-t):
    #         dt = params.MaxTime()-t
    #     t += dt
    #     u = ti.Evolve(dt,t,u)
    #     nsteps += 1
    #     if (nsteps % 100 == 0 and params.PlotSol()==True): 
    #         print("time = %1.4f" %t, "dt = %1.3e"%dt)
    #         PlotSolution(t, mesh.X(), u, fig)

    # if (params.PlotSol() == True):  
    #     PlotSolution(t, mesh.X(), u, fig)
    #     plt.show()
