from sympy import Symbol, Eq, Function, tanh, sin, cos, sqrt
import tensorflow as tf
import numpy as np

from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain, MonitorDomain, InferenceDomain
from modulus.data import Validation, Monitor, BC, Inference
from modulus.sympy_utils.geometry_2d import Rectangle, Circle, Line
from modulus.PDES.navier_stokes import NavierStokes
from modulus.controller import ModulusController
from modulus.variables import Variables, Key
from modulus.pdes import PDES
from modulus.node import Node

"""
# params for domain
height = 0.2
width = 0.4
radius = 0.02
circle_pos = (-width/4, 0)
vel = 1.0
boundary = ((-width / 2, -height / 2), (width / 2, height / 2))
bounds_x = (-width / 2, width / 2)
bounds_y = (-height / 2, height / 2)

# fluid params
nu = 4.0e-4
"""
# params for domain
height = 2.0
width = 4.0
radius = 0.2
circle_pos = (-width / 4, 0)
vel = 1.0
boundary = ((0, 0), (width, height))
bounds_x = (0, width)
bounds_y = (0, height)

# fluid params
nu = 4.0e-3

re = int((radius * 2) / nu)  # Reynolds Number

# define geometry
rec = Rectangle(boundary[0], boundary[1])
circle = Circle(circle_pos, radius)
geo = rec - circle

"""
# Continuity lines
plane1 = Line((boundary[0][0]+0.1, boundary[0][1]),(boundary[0][0]+0.1, boundary[1][1]), 1)
plane2 = Line((boundary[0][0]+0.2, boundary[0][1]),(boundary[0][0]+0.2, boundary[1][1]), 1)
plane3 = Line((boundary[0][0]+0.3, boundary[0][1]),(boundary[0][0]+0.3, boundary[1][1]), 1)
plane4 = Line((boundary[0][0]+0.4, boundary[0][1]),(boundary[0][0]+0.4, boundary[1][1]), 1)
"""
# define sympy variables to parametrize domain curves
x, y = Symbol('x'), Symbol('y')

# param range
total_nr_iterations = 20

# time window size
time_window_size = 30 / total_nr_iterations  # TODO total_nr_iterations - 1? Depending on if the initial is one window
                                             # TODO or if iteration 0000 is the first time window

# time domain
t_symbol = Symbol('t')
time_range = (0, time_window_size)
param_ranges = {t_symbol: time_range}


class ICTrain(TrainDomain):  # TODO change all bc to fit with new and old bounds
    name = 'initial_conditions'
    nr_iterations = 1

    def __init__(self, **config):
        super(ICTrain, self).__init__()
        batch_size = 256

        ic = geo.interior_bc(outvar_sympy={'u': 0,
                                           'v': 0,
                                           'p': 0},  # TODO set correct pressure equation and batch_size
                             batch_size_per_area=batch_size * 8,
                             bounds={x: bounds_x,
                                     y: bounds_y},
                             lambda_sympy={'lambda_u': 100,
                                           'lambda_v': 100,
                                           'lambda_p': 100},
                             param_ranges={t_symbol: 0})
        self.add(ic, name="ic")

        # left wall inlet
        leftWall = rec.boundary_bc(outvar_sympy={'u': vel, 'v': 0},
                                   batch_size_per_area=batch_size,
                                   lambda_sympy={'lambda_u': 1.0 - ((2 * abs(y)) / height),  # weight edges to be zero
                                                 'lambda_v': 1.0},
                                   criteria=Eq(x, -width / 2), # TODO change with the desktop variant for all...
                                   param_ranges=param_ranges)
        self.add(leftWall, name="leftWall")

        # no slip top wall
        topWall = rec.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                  batch_size_per_area=batch_size,
                                  criteria=Eq(y, height / 2),
                                  param_ranges=param_ranges)
        self.add(topWall, name="topWallNoSlip")

        # no slip bottom wall
        bottomWall = rec.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                     batch_size_per_area=batch_size,
                                     criteria=Eq(y, -height / 2),
                                     param_ranges=param_ranges)
        self.add(bottomWall, name="bottomWallNoSlip")

        # circle no slip
        circleBC = circle.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                      batch_size_per_area=batch_size,
                                      param_ranges=param_ranges)
        self.add(circleBC, name="circleNoSlip")

        # right wall outlet 0 pressure
        rightWall = rec.boundary_bc(outvar_sympy={'p': 0},
                                    batch_size_per_area=batch_size,
                                    criteria=Eq(x, width / 2),
                                    param_ranges=param_ranges)
        self.add(rightWall, name="rightWall")

        # interior
        interior = geo.interior_bc(outvar_sympy={'continuity': 0, 'momentum_x': 0, 'momentum_y': 0},
                                   bounds={x: bounds_x,
                                           y: bounds_y},
                                   lambda_sympy={'lambda_continuity': geo.sdf,
                                                 'lambda_momentum_x': geo.sdf,
                                                 'lambda_momentum_y': geo.sdf},
                                   # criteria=(sqrt((x - circle_pos[0])**2 + (y - circle_pos[1])**2) > radius) ,
                                   # criteria=((x - circle_pos[0])**2 + (y - circle_pos[1])**2) > radius**2,
                                   batch_size_per_area=batch_size * 8,
                                   param_ranges=param_ranges)
        self.add(interior, name="Interior")


class IterativeTrain(TrainDomain):
    name = 'iteration'
    nr_iterations = total_nr_iterations - 1

    def __init__(self, **config):
        super(IterativeTrain, self).__init__()
        batch_size = 256
        ic = geo.interior_bc(outvar_sympy={'u_ic': 0,
                                           'v_ic': 0,
                                           'p_ic': 0},  # TODO set correct batch_size
                             batch_size_per_area=batch_size * 8,
                             bounds={x: bounds_x,
                                     y: bounds_y},
                             lambda_sympy={'lambda_u_ic': 100,
                                           'lambda_v_ic': 100,
                                           'lambda_p_ic': 100},
                             param_ranges={t_symbol: 0})
        self.add(ic, name="IterativeIC")

        # left wall inlet
        leftWall = rec.boundary_bc(outvar_sympy={'u': vel, 'v': 0},
                                   batch_size_per_area=batch_size,
                                   lambda_sympy={'lambda_u': 1.0 - ((2 * abs(y)) / height),
                                                 # weight edges to be zero TODO set same as dolfin maybe
                                                 'lambda_v': 1.0},
                                   criteria=Eq(x, -width / 2), # TODO change with the desktop variant for all...
                                   param_ranges=param_ranges)
        self.add(leftWall, name="IterativeleftWall")

        # no slip top wall
        topWall = rec.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                  batch_size_per_area=batch_size,
                                  criteria=Eq(y, height / 2),
                                  param_ranges=param_ranges)
        self.add(topWall, name="IterativetopWallNoSlip")

        # no slip bottom wall
        bottomWall = rec.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                     batch_size_per_area=batch_size,
                                     criteria=Eq(y, -height / 2),
                                     param_ranges=param_ranges)
        self.add(bottomWall, name="IterativebottomWallNoSlip")

        # circle no slip
        circleBC = circle.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                      batch_size_per_area=batch_size,
                                      param_ranges=param_ranges)
        self.add(circleBC, name="IterativecircleNoSlip")

        # right wall outlet 0 pressure
        rightWall = rec.boundary_bc(outvar_sympy={'p': 0},
                                    batch_size_per_area=batch_size,
                                    criteria=Eq(x, width / 2),
                                    param_ranges=param_ranges)
        self.add(rightWall, name="IterativerightWall")

        # interior
        interior = geo.interior_bc(outvar_sympy={'continuity': 0, 'momentum_x': 0, 'momentum_y': 0},
                                   bounds={x: bounds_x,
                                           y: bounds_y},
                                   lambda_sympy={'lambda_continuity': geo.sdf,  # TODO really best weights for this?
                                                 'lambda_momentum_x': geo.sdf,
                                                 'lambda_momentum_y': geo.sdf},
                                   # criteria=(sqrt((x - circle_pos[0])**2 + (y - circle_pos[1])**2) > radius) ,
                                   # criteria=((x - circle_pos[0])**2 + (y - circle_pos[1])**2) > radius**2,
                                   batch_size_per_area=batch_size * 8,
                                   param_ranges=param_ranges)
        self.add(interior, name="IterativeInterior")


class VKVSInference(InferenceDomain):
    def __init__(self, **config):
        super(VKVSInference, self).__init__()
        # inf data time 0
        res = 80
        mesh_x, mesh_y = np.meshgrid(np.linspace(bounds_x[0], bounds_x[1], res),
                                     # TODO fix ranges and expand_dims(mesh.flatten())
                                     np.linspace(bounds_y[0], bounds_y[1], res),
                                     indexing='ij')
        mesh_x = np.expand_dims(mesh_x.flatten(), axis=-1)
        mesh_y = np.expand_dims(mesh_y.flatten(), axis=-1)
        for i, specific_t in enumerate(np.linspace(time_range[0], time_window_size, 5)):
            interior = {'x': mesh_x,
                        'y': mesh_y,
                        't': np.full_like(mesh_x, specific_t)}
            inf = Inference(interior, ['u', 'v', 'p', 'shifted_t'])
            self.add(inf, "Inference_" + str(i).zfill(4))

        # TODO Easier way?
        interior = geo.sample_interior(1e3, bounds={x: bounds_x, y: bounds_y})
        for i, specific_t in enumerate(np.linspace(time_range[0], time_window_size, 5)):
            interior['t'] = np.full_like(mesh_x, specific_t)
            inf = Inference(interior, ['u', 'v', 'p', 'shifted_t'])
            self.add(inf, "Inference_" + str(i).zfill(4))
        # interior = Inference(geo.sample_interior(1e5, bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)}),
        #                     ['u', 'v', 'p'])
        """
        # inf data time 0
        res = 256
        mesh_x, mesh_y = np.meshgrid(np.linspace(0, 2*np.pi, res),
                                     np.linspace(0, 2*np.pi, res),
                                     indexing='ij')
        mesh_x = np.expand_dims(mesh_x.flatten(), axis=-1)
        mesh_y = np.expand_dims(mesh_y.flatten(), axis=-1)
        for i, specific_t in enumerate(np.linspace(time_range[0], time_window_size, 5)):
          interior = {'x': mesh_x,
                      'y': mesh_y,
                      't': np.full_like(mesh_x, specific_t)}
          inf = Inference(interior, ['u','v','p','shifted_t'])
          self.add(inf, "InferencePlane_"+str(i).zfill(4))
        """


class VKVSSolver(Solver):
    seq_train_domain = [ICTrain, IterativeTrain]
    iterative_train_domain = IterativeTrain
    inference_domain = VKVSInference

    def __init__(self, **config):
        super(VKVSSolver, self).__init__(**config)

        # make time window that moves
        self.time_window = tf.get_variable("time_window", [],
                                           initializer=tf.constant_initializer(0),
                                           trainable=False,
                                           dtype=tf.float32)

        def slide_time_window(invar):
            outvar = Variables()
            outvar['shifted_t'] = invar['t'] + self.time_window
            print(self.time_window)  # TODO idea to see what time we are computing
            return outvar

        # make node for difference between velocity and the previous time window of velocity
        def make_ic_loss(invar):
            outvar = Variables()
            outvar['u_ic'] = invar['u'] - tf.stop_gradient(invar['u_prev_step'])
            outvar['v_ic'] = invar['v'] - tf.stop_gradient(invar['v_prev_step'])
            # outvar['w_ic'] = invar['w'] - tf.stop_gradient(invar['w_prev_step'])
            outvar['p_ic'] = invar['p'] - tf.stop_gradient(invar['p_prev_step'])
            return outvar

        """
        # make node periodic boundary
        def make_periodic_boundary(invar):
          outvar = Variables() 
          outvar['x_sin'] = tf.sin(invar['x'])
          outvar['x_cos'] = tf.cos(invar['x'])
          outvar['y_sin'] = tf.sin(invar['y'])
          outvar['y_cos'] = tf.cos(invar['y'])
          outvar['z_sin'] = tf.sin(invar['z'])
          outvar['z_cos'] = tf.cos(invar['z'])
          return outvar
        """

        self.equations = (NavierStokes(nu=nu, rho=1, dim=2, time=True).make_node()
                          # + [Node(make_periodic_boundary)]
                          + [Node(make_ic_loss)]
                          + [Node(slide_time_window)])

        flow_net = self.arch.make_node(name='flow_net',
                                       inputs=[  # 'x_sin', 'x_cos',
                                           # 'y_sin', 'y_cos',
                                           # 'z_sin', 'z_cos',
                                           'x', 'y',
                                           'shifted_t'],
                                       outputs=['u',
                                                'v',
                                                # 'w',
                                                'p'])
        flow_net_prev_step = self.arch.make_node(name='flow_net_prev_step',
                                                 inputs=[  # 'x_sin', 'x_cos',
                                                     # 'y_sin', 'y_cos',
                                                     # 'z_sin', 'z_cos',
                                                     'x', 'y',
                                                     'shifted_t'],
                                                 outputs=['u_prev_step',
                                                          'v_prev_step',
                                                          # 'w_prev_step',
                                                          'p_prev_step'])
        self.nets = [flow_net, flow_net_prev_step]

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

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'network_dir': './network_checkpoint_vkvs_w_re' + str(re),
            'layer_size': 256,  # TODO change layer size
            'max_steps': 10000,
            'decay_steps': 3000,  # TODO check what this is, lr??
            'xla': True,
        })


if __name__ == '__main__':
    ctr = ModulusController(VKVSSolver)
    ctr.run()
