from sympy import Symbol, Eq, Function, tanh, sin, cos, sqrt
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
from modulus.architecture import ModifiedFourierNetArch

# params for domain
height = 2.0
width = 4.0
radius = 0.2
circle_pos = (width / 4, height / 2)
vel = 1.0
boundary = ((0, 0), (width, height))
bounds_x = (0, width)
bounds_y = (0, height)
#max_distance = 2.0  # TODO check if this is correct

# fluid params
nu = 4.0e-3

re = int((radius * 2) / nu)  # Reynolds Number

# define geometry
rec = Rectangle(boundary[0], boundary[1])
circle = Circle(circle_pos, radius)
geo = rec - circle

# define sympy variables to parametrize domain curves
x, y = Symbol('x'), Symbol('y')

# param range
#total_nr_iterations = 12

# time window size
#time_window_size = 6 / total_nr_iterations  # TODO total_nr_iterations - 1? Depending on if the initial is one window
# TODO or if iteration 0000 is the first time window

# time domain
t_symbol = Symbol('t')
#time_range = (0, time_window_size)
time_range = (0, 12)
param_ranges = {t_symbol: time_range}

directory = './vkvs_newnew_re' + str(re) + '_t_' + str(time_range[0]) + '-' + str(time_range[1])  # Results directory


class VKVSTrain(TrainDomain):
    #name = 'initial_conditions'
    #nr_iterations = 1

    def __init__(self, **config):
        super(VKVSTrain, self).__init__()
        batch_size = 64

        ic = geo.interior_bc(outvar_sympy={'u': 0,
                                           'v': 0,
                                           'p': 0},
                             batch_size_per_area=batch_size * 8,
                             bounds={x: bounds_x,
                                     y: bounds_y},
                             lambda_sympy={'lambda_u': 100,
                                           'lambda_v': 100,
                                           'lambda_p': 100},
                             param_ranges={t_symbol: 0},
                             quasirandom=True)
        self.add(ic, name="ic")

        # left wall inlet
        leftWall = rec.boundary_bc(outvar_sympy={'u': vel, 'v': 0},
                                   batch_size_per_area=batch_size,
                                   lambda_sympy={'lambda_u': 1.0 - ((2.0 * abs(y - 1.0)) / 2.0),
                                                 'lambda_v': 1.0},
                                   criteria=Eq(x, bounds_x[0]),
                                   param_ranges=param_ranges,
                                   quasirandom=True)
        self.add(leftWall, name="leftWall")

        # no slip top wall
        topWall = rec.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                  batch_size_per_area=batch_size,
                                  criteria=Eq(y, bounds_y[1]),
                                  param_ranges=param_ranges,
                                  quasirandom=True)
        self.add(topWall, name="topWallNoSlip")

        # no slip bottom wall
        bottomWall = rec.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                     batch_size_per_area=batch_size,
                                     criteria=Eq(y, bounds_y[0]),
                                     param_ranges=param_ranges,
                                     quasirandom=True)
        self.add(bottomWall, name="bottomWallNoSlip")

        # circle no slip
        circleBC = circle.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                      batch_size_per_area=batch_size,
                                      param_ranges=param_ranges,
                                      quasirandom=True)
        self.add(circleBC, name="circleNoSlip")

        # right wall outlet 0 pressure
        rightWall = rec.boundary_bc(outvar_sympy={'p': 0},
                                    batch_size_per_area=batch_size,
                                    criteria=Eq(x, bounds_x[1]),
                                    param_ranges=param_ranges,
                                    quasirandom=True)
        self.add(rightWall, name="rightWall")

        # interior
        interior = geo.interior_bc(outvar_sympy={'continuity': 0, 'momentum_x': 0, 'momentum_y': 0},
                                   bounds={x: bounds_x,
                                           y: bounds_y},
                                   lambda_sympy={'lambda_continuity': geo.sdf, # TODO test without sdf weighting
                                                 'lambda_momentum_x': geo.sdf,
                                                 'lambda_momentum_y': geo.sdf},
                                   batch_size_per_area=batch_size * 8 * 2,
                                   param_ranges=param_ranges,
                                   quasirandom=True)
        self.add(interior, name="Interior")


class VKVSInference(InferenceDomain): # TODO convert to without time windows...
    def __init__(self, **config):
        super(VKVSInference, self).__init__()
        # inf data time 0
        """
        res = 128
        mesh_x, mesh_y = np.meshgrid(np.linspace(bounds_x[0], bounds_x[1], res),
                                     np.linspace(bounds_y[0], bounds_y[1], res),
                                     indexing='ij')
        mesh_x = np.expand_dims(mesh_x.flatten(), axis=-1)
        mesh_y = np.expand_dims(mesh_y.flatten(), axis=-1)
        """
        mesh = geo.sample_interior(1e3, bounds={x: bounds_x, y: bounds_y})
        for i, specific_t in enumerate(np.linspace(time_range[0], time_range[1], 10)):
            #interior = {'x': mesh_x,
            #            'y': mesh_y,
            #            't': np.full_like(mesh_x, specific_t)}
            interior2 = {'x': mesh['x'],
                         'y': mesh['y'],
                         't': np.full_like(mesh['x'], specific_t)}
            #print("DEBUG INFERENCE: " + str(specific_t))
            #print("DEBUG INFERENCE CORRECT: " + str(len(interior['x'])) + "," + str(len(interior['x'])) + ", " + str(len(interior['t'])))
            #inf = Inference(interior, ['u', 'v', 'p', 't'])
            inf2 = Inference(interior2, ['u', 'v', 'p', 't'])
            #self.add(inf, "Inference_" + str(i).zfill(4)) 
            self.add(inf2, "NewInference_" + str(i).zfill(4))
        """
        interior2 = geo.sample_interior(1e3, bounds={x: bounds_x, y: bounds_y})
        #print("DEBUG INFERENCE: " + str(len(interior2['x'])) + "," + str(len(interior2['x'])))
        for i, specific_t in enumerate(np.linspace(time_range[0], time_range[1], time_range[1])):
            interior2['t'] = np.full_like(interior2['x'], specific_t)  # TODO time does not work correctly
            #print("DEBUG INFERENCE: " + str(len(interior2['t'])))
            inf2 = Inference(interior2, ['u', 'v', 'p', 't'])
            self.add(inf2, "NewInference_" + str(i).zfill(4))
        """


class VKVSSolver(Solver):
    #seq_train_domain = [ICTrain, IterativeTrain]
    #iterative_train_domain = IterativeTrain
    train_domain = VKVSTrain
    inference_domain = VKVSInference
    arch = ModifiedFourierNetArch
    convergence_check = 1.0e-30

    def __init__(self, **config):
        super(VKVSSolver, self).__init__(**config)

        """
        # make time window that moves
        self.time_window = tf.get_variable("time_window", [],
                                           initializer=tf.constant_initializer(0),
                                           trainable=False,
                                           dtype=tf.float32)
        
        def slide_time_window(invar):
            outvar = Variables()
            outvar['shifted_t'] = invar['t'] + self.time_window
            return outvar

        # make node for difference between velocity and the previous time window of velocity
        def make_ic_loss(invar):
            outvar = Variables()
            outvar['u_ic'] = invar['u'] - tf.stop_gradient(invar['u_prev_step'])
            outvar['v_ic'] = invar['v'] - tf.stop_gradient(invar['v_prev_step'])
            outvar['p_ic'] = invar['p'] - tf.stop_gradient(invar['p_prev_step'])
            return outvar
        """

        self.equations = (NavierStokes(nu=nu, rho=1.0, dim=2, time=True).make_node()
                          # + KEpsilon(nu=nu, rho=1, dim=2, time=True).make_node()
                          # + ZeroEquation(nu=nu, dim=2, time=True, max_distance=max_distance).make_node()
                          # + [Node.from_sympy(geo.sdf, 'normal_distance')]
                          #+ [Node(make_ic_loss)]
                          #+ [Node(slide_time_window)])
                          )

        flow_net = self.arch.make_node(name='flow_net',
                                       inputs=['x', 'y',
                                               't'], #'shifted_t'],
                                       outputs=['u',
                                                'v',
                                                'p'])
        """
        flow_net_prev_step = self.arch.make_node(name='flow_net_prev_step',
                                                 inputs=['x', 'y',
                                                         'shifted_t'],
                                                 outputs=['u_prev_step',
                                                          'v_prev_step',
                                                          'p_prev_step'])
        self.nets = [flow_net, flow_net_prev_step]
        """
        self.nets = [flow_net]
    """
    def custom_update_op(self):
        # zero train step op
        global_step = [v for v in tf.get_collection(tf.GraphKeys.VARIABLES) if 'global_step' in v.name][0]
        zero_step_op = tf.assign(global_step, tf.zeros_like(global_step))

        # make update op that shifts time window
        update_time = tf.assign_add(self.time_window, time_window_size)

        # make update op that sets weights from_flow_net to flow_net_prev_step
        prev_assign_step = []
        flow_net_variables = [v for v in tf.trainable_variables() if 'flow_net/' in v.name]
        flow_net_prev_step_variables = [v for v in tf.trainable_variables() if 'flow_net_prev_step' in v.name]
        for v, v_prev_step in zip(flow_net_variables, flow_net_prev_step_variables):
            prev_assign_step.append(tf.assign(v_prev_step, v))
        prev_assign_step = tf.group(*prev_assign_step)

        return tf.group(update_time, zero_step_op, prev_assign_step)
        """
    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'network_dir': directory,
            'layer_size': 256,
            'max_steps': 3000000,
            'decay_steps': 30000,
            'xla': True,
            'adaptive_activations': True,
            'save_filetypes': 'vtk, np'
            #'convergence_check': 5.0e-3
        })


if __name__ == '__main__':
    try:
        mkdir(directory)
    except:
        print("Directory already exists")

    f = open(directory + '/benchmark_data.txt', 'w')
    ctr = ModulusController(VKVSSolver)
    bench_start_time = perf_counter()
    ctr.run()
    bench_end_time = perf_counter()
    f.write("Elapsed time: " + str(bench_end_time - bench_start_time))
    f.close()
    print("Elapsed time: " + str(bench_end_time - bench_start_time))
