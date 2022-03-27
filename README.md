# Discontinuous Galerkin Code
# One Dimension

This solves one dimension nonlinear hyperbolic conservation system. Currently there is a Burgers' equation and Euler's equation implemented. Euler has the Sod Shock tube problem. 

If other initial conditions are desired then please look at `problems.py` to see how to set up different initial conditions.

In order to set up a new set of equations then please look at `euler.py` to see how to implement the class that takes care of the physics and fluxes.
Only depends on Numpy. 

If you have any questions, please feel to email me at mrodrig6 at ucsc dot edu.
