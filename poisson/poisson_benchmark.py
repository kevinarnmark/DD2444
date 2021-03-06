from sympy import Symbol, Eq, Function, tanh, sin, cos, sqrt
from time import perf_counter
from os import mkdir
import tensorflow as tf
import numpy as np

from newSolver import Solver2 as Solver
#from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain, MonitorDomain, InferenceDomain
from modulus.data import Validation, Monitor, BC, Inference
from modulus.sympy_utils.geometry_2d import Rectangle
from modulus.controller import ModulusController
from modulus.variables import Variables, Key
from modulus.pdes import PDES
from modulus.node import Node
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

directory = './poisson_bench_network_checkpoint'  # Results directory

class PoissonSmooth(PDES):
    name="Poisson Equation"

    def __init__(self):
        # set params
        #self.u = u
        #self.dim = dim

        # coordinates
        x, y = Symbol('x'), Symbol('y')

        # make input variables
        input_variables = {'x': x, 'y': y}

        # Scalar function
        #assert type(u) == str, "u needs to be string"
        u = Function('u')(*input_variables)

        # Smooth Source
        f = (1 / 4)*sum((-1)**(k+1)*2*k*sin(k*np.pi*x)*sin(k*np.pi*y) for k in range(1,5))

        # set equations
        self.equations = Variables()
        self.equations['poisson_equation'] = ( f
                                           - u.diff(x, 2)
                                           - u.diff(y, 2))
        #self.equations['poisson_equation'] = ((u.diff(x, 2) + u.diff(y, 2)) - f)


class PoissonTrain(TrainDomain):
    #name = 'iteration'
    #nr_iterations = total_nr_iterations - 1

    def __init__(self, **config):
        super(PoissonTrain, self).__init__()
        #self.batch_size = batch_size
        # boundary
        boundary_condition = geo.boundary_bc(outvar_sympy={'u': 0},
                                    batch_size_per_area=1000,
                                    #criteria=Eq(x, bounds_x[1]),
                                    #param_ranges=param_ranges,
                                    quasirandom=False)
        self.add(boundary_condition, name="BC")

        # interior
        interior = geo.interior_bc(outvar_sympy={'poisson_equation': 0},
                                   bounds={x: bounds_x,
                                           y: bounds_y},
                                   lambda_sympy={'lambda_poisson_equation': 1.0},
                                   batch_size_per_area=128**2,
                                   #param_ranges=param_ranges,
                                   quasirandom=False)
        self.add(interior, name="Interior")


class PoissonInference(InferenceDomain):
    def __init__(self, **config):
        super(PoissonInference, self).__init__()

        mesh = geo.sample_interior(1e4, bounds={x: bounds_x, y: bounds_y})

        inf = Inference(mesh, ['u'])
        self.add(inf, "Inference")


class PoissonSolverBase(Solver):
    #seq_train_domain = [ICTrain, IterativeTrain]
    train_domain = PoissonTrain
    inference_domain = PoissonInference
    #arch = ModifiedFourierNetArch
    convergence_check = 1.0e-6
    """
    def __init__(self, **config):
        super(PoissonSolverBase, self).__init__(**config)

        self.equations = (PoissonSmooth().make_node())

        poisson_net = self.arch.make_node(name='poisson_net',
                                       inputs=['x', 'y'],
                                       outputs=['u'])
        self.nets = [poisson_net]

        self.save_network_freq = 5000
        self.print_stats_freq = 500
        self.tf_summary_freq = 500

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'network_dir': directory,
            'layer_size': 256,
            #'nr_layer': 8,
            'max_steps': 5000,
            'decay_steps': 2500,
            'xla': True,
            'adaptive_activations': False,
            'save_filetypes': 'vtk, np'
            #'convergence_check': 5.0e-3
        })

    """

def __constructor__(self, **config):
    # initialize super of class type
    super(self.__class__, self).__init__(**config)

    self.equations = (PoissonSmooth().make_node())

    poisson_net = self.arch.make_node(name='poisson_net',
                                       inputs=['x', 'y'],
                                       outputs=['u'])
    self.nets = [poisson_net]

    self.save_network_freq = 5000
    self.print_stats_freq = 500
    self.tf_summary_freq = 500


if __name__ == '__main__':
    try:
        mkdir(directory)
    except:
        print("Directory already exists")

    f = open(directory + '/benchmark_data.txt', 'w')

    archs = [FullyConnectedArch, ModifiedFourierNetArch]
    
    for arch in archs:
        try:
            directory_name = directory+'/'+arch.__name__
            mkdir(directory_name)
        except:
            print("Directory already exists")
        
        @classmethod
        def update_defaults(cls, defaults):
            defaults.update({
                'network_dir': directory_name,
                'layer_size': 256,
                #'nr_layer': 8,
                'max_steps': 1000,
                'decay_steps': 500,
                'xla': True,
                'adaptive_activations': False,
                'save_filetypes': 'vtk, np'
                #'convergence_check': 5.0e-3
            })

        # Dynamically creating class to change parameters
        PoissonSolver2 = type("PoissonSolver", (Solver, ), {
            "__init__": __constructor__,

            "train_domain": PoissonTrain,
            "inference_domain": PoissonInference,
            "arch": arch,
            "convergence_check": 1.0e-6,

            "update_defaults": update_defaults

        })

        ctr = ModulusController(PoissonSolver2)
        bench_start_time = perf_counter()
        ctr.run()
        bench_end_time = perf_counter()
        f.write(arch.__name__ + "Elapsed time: " + str(bench_end_time - bench_start_time))
        print(arch.__name__ + "Elapsed time: " + str(bench_end_time - bench_start_time))
    
    f.close()
    """
    ctr = ModulusController(PoissonSolver)
    bench_start_time = perf_counter()
    ctr.run()
    bench_end_time = perf_counter()
    f.write("Elapsed time: " + str(bench_end_time - bench_start_time))
    f.close()
    print("Elapsed time: " + str(bench_end_time - bench_start_time))
    """
