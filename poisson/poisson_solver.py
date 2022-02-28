#from sympy import Symbol, Eq, Function, tanh, sin, cos, sqrt
from sympy import *
from sympy.vector import laplacian
from time import perf_counter
from os import mkdir
import tensorflow as tf
import numpy as np

from newSolver import Solver2 as Solver  # from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain, MonitorDomain, InferenceDomain
from modulus.data import Validation, Monitor, BC, Inference
from modulus.sympy_utils.geometry_2d import Rectangle, Circle, Line
from modulus.PDES import NavierStokes, ZeroEquation, KEpsilon
from modulus.controller import ModulusController
from modulus.variables import Variables, Key
from modulus.pdes import PDES
from modulus.node import Node
from modulus.architecture.radial_basis import RadialBasisArch
from modulus.architecture import *

# params for domain
height = 1.0  # 2.0
width = 1.0  # 4.0
boundary = ((0, 0), (width, height))
bounds_x = (0, width)
bounds_y = (0, height)

# define geometry
rec = Rectangle(boundary[0], boundary[1])
geo = rec #- circle

# define sympy variables to parametrize domain curves
x, y = Symbol('x'), Symbol('y')

# Sample batch size
batch_size = 2**10

directory = './poisson_network_checkpoint_fully_connected'  # Results directory

class PoissonSmooth(PDES):
    name="Poisson Equation"

    def __init__(self):
        # coordinates
        x, y = Symbol('x'), Symbol('y')

        # make input variables
        input_variables = {'x': x, 'y': y}

        # Scalar function
        u = Function('u')(*input_variables)

        # Smooth Source
        f = -sum(((-1)**(k+1))*2*k*sin(k*np.pi*x)*sin(k*np.pi*y) for k in range(1,5))
        # set equations
        self.equations = Variables()

        self.equations['poisson_equation'] = (f - (u.diff(x, 2) + u.diff(y, 2)))


class PoissonTrain(TrainDomain):

    def __init__(self, **config):
        super(PoissonTrain, self).__init__()

        boundary_condition = geo.boundary_bc(outvar_sympy={'u': 0.0},
                                    batch_size_per_area=1000)
        self.add(boundary_condition, name="BC")

        # interior
        interior = geo.interior_bc(outvar_sympy={'poisson_equation': 0.0},
                                   bounds={x: bounds_x,
                                           y: bounds_y},
                                   batch_size_per_area=128**2)
        self.add(interior, name="Interior")


class PoissonInference(InferenceDomain):
    def __init__(self, **config):
        super(PoissonInference, self).__init__()
        deltaX = 0.001
        deltaY = 0.001
        x = np.arange(0, 1, deltaX)
        y = np.arange(0, 1, deltaY)
        X, Y = np.meshgrid(x, y)
        X = np.expand_dims(X.flatten(), axis=-1)
        Y = np.expand_dims(Y.flatten(), axis=-1)
        mesh = {'x': X,
                'y': Y}

        inf = Inference(mesh, ['u'])
        self.add(inf, "Inference")


class PoissonVal(ValidationDomain):
    def __init__(self, **config):
        super(PoissonVal, self).__init__()

        deltaX = 0.01
        deltaY = 0.01
        x = np.arange(0, 1, deltaX)
        y = np.arange(0, 1, deltaY)
        X, Y = np.meshgrid(x, y)
        X = np.expand_dims(X.flatten(), axis=-1)
        Y = np.expand_dims(Y.flatten(), axis=-1)
        #u = (1 / 4)*sum((-1)**(k+1)*2*k*np.sin(k*np.pi*X)*np.sin(k*np.pi*Y) for k in range(1,5))
        #u = (1 / 4)*sum((-2/(k*np.pi**2))*(-1)**(k+1)*np.sin(k*np.pi*X)*np.sin(k*np.pi*Y) for k in range(1,5))
        invar_numpy = {'x': X, 'y': Y}
        outvar_numpy = {'u': u}
        val = Validation.from_numpy(invar_numpy, outvar_numpy)
        self.add(val, name='Val')


class PoissonSolver(Solver):
    train_domain = PoissonTrain
    inference_domain = PoissonInference
    arch = FullyConnectedArch
    convergence_check = 1.0e-8

    def __init__(self, **config):
        super(PoissonSolver, self).__init__(**config)

        self.equations = (PoissonSmooth().make_node())

        poisson_net = self.arch.make_node(name='poisson_net',
                                       inputs=['x', 'y'],
                                       outputs=['u'])
        self.nets = [poisson_net]

        self.save_network_freq = 500
        self.print_stats_freq = 500
        self.tf_summary_freq = 500

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'network_dir': directory,
            'layer_size': 256,
            'nr_layers': 8,
            'max_steps': 20000,
            'decay_steps': 10000,
            'xla': True,
            'save_filetypes': 'vtk, np'
        })




if __name__ == '__main__':
    try:
        mkdir(directory)
    except:
        print("Directory already exists")

    f = open(directory + '/benchmark_data.txt', 'w')
    ctr = ModulusController(PoissonSolver)
    bench_start_time = perf_counter()
    ctr.run()
    bench_end_time = perf_counter()
    f.write("Elapsed time: " + str(bench_end_time - bench_start_time))
    f.close()
    print("Elapsed time: " + str(bench_end_time - bench_start_time))