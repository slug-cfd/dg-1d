import numpy as np
import linedg as ldg
import SimulationParameters as sp

if __name__ == "__main__":

    params = sp.Parameters(2,3,30,np.array([0,1]),0.8,0.25,"rk4","periodic","godunov","False")
    interp = ip.Interpolation(params.Nnodes(), params.Nquads())
    m      = mesh.Mesh(params.Domain(), params.Nels(), params.Nnodes(), params.Nquads())

    