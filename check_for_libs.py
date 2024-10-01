

# from acados import *


import PyQt5

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

plt.plot([1,2,3], [1,0,2])
plt.show()


'''

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, AcadosSimSolver, AcadosSim
import numpy as np
# from casadi import SX, vertcat, diag
from casadi import *
from typing import Tuple


def test_casadi():
    x = MX.sym('x',2); # Two states

    # Expression for ODE right-hand side
    z = 1-x[1]**2
    rhs = vertcat(z*x[0]-x[1],x[0])

    ode = {}         # ODE declaration
    ode['x']   = x   # states
    ode['ode'] = rhs # right-hand side

    # Construct a Function that integrates over 4s
    F = integrator('F','cvodes',ode,0,4)

    # Start from x=[0;1]
    res = F(x0=[0,1])

    print(res["xf"])

    # Sensitivity wrt initial state
    res = F(x0=x)
    S = Function('S',[x],[jacobian(res["xf"],x)])
    print(S([0,1]))

def test_numpy():
    array = np.array([[1, 2], [3, 4]])
    print(array)

def test_matplotlib():
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.title("Sample plot")
    plt.show()


# test_casadi()
# test_numpy()
test_matplotlib()

import sys
print(sys.executable)

'''