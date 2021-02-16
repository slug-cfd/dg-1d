import numpy as np
import matplotlib.pyplot as plt
import Interpolation as Interpolation
import SimulationParameters as sp
import problems as pr
import Mesh as m
import time_int as tint
import linedg
import Limit
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
import sys
np.set_printoptions(linewidth=250)



def PlotSolution(t, x, u, fig):
    xplot = x.ravel()
    uplot = u.ravel()
    fig.clf()
    plt.plot(xplot,uplot,'b-')
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Time = %1.3f" %t)
    plt.grid()
    plt.pause(0.05)
    plt.draw()


if __name__ == "__main__":



    
    probs = pr.problem(3)
    params, mesh, u = probs.SetProblem()

    print("=================================================================")
    print("*                    Simulation                                 *")
    print("*                     DG Code                                   *")
    print("*             Order = ", params.order() )
    print("*         Number of Elements = ", params.nels())
    print("*            CFL Constant = ", params.CourantNumber())
    print("*              Problem  = ", probs.GetProblem())
    print("=================================================================")

    # params.UpdatePlotSol(False)
    utest = np.ones([params.nels(), params.nnodes()])



    ldg = linedg.linedg(mesh, params)
    ti = tint.time_integration(ldg)
    # lim = Limit.Limit(ldg)
    # ubar = lim.LimitSolution(u)

    t = 0.0
    nsteps = 0
    fig = plt.figure()
    while t < params.MaxTime():
        dt = ti.ComputeDt(u)
        if (dt > params.MaxTime()-t):
            dt = params.MaxTime()-t
        t += dt
        u = ti.Evolve(dt,t,u)
        nsteps += 1
        if (nsteps % 100 == 0 and params.PlotSol()==True): 
            print("time = %1.4f" %t, "dt = %1.3e"%dt)
            PlotSolution(t, mesh.X(), u, fig)

    if (params.PlotSol() == True):  
        PlotSolution(t, mesh.X(), u, fig)
        plt.show()
