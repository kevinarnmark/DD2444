import tensorflow as tf

from sympy import Symbol, Eq, Abs

from modulus.solver import Solver
from modulus.dataset import TrainDomain, ValidationDomain, InferenceDomain, MonitorDomain
from modulus.data import Validation, Inference, Monitor
from modulus.sympy_utils.geometry_2d import Rectangle, Circle, Line
from modulus.csv_utils.csv_rw import csv_to_dict
from modulus.PDES import NavierStokes, IntegralContinuity
from modulus.controller import ModulusController
from math import sqrt

# TODO add time, add criteria (relative error), change network size and types and such, compare to fenics


# params for domain
height = 0.2
width = 0.4
radius = 0.02
circle_pos = (-width/4, 0)
vel = 1.0
boundary = ((-width / 2, -height / 2), (width / 2, height / 2))

# fluid params
viscosity = 4.0e-4


re = int((radius*2)/viscosity) # Reynolds Number


# define geometry
rec = Rectangle(boundary[0], boundary[1])
circle = Circle(circle_pos, radius)
geo = rec - circle

# Continuity lines
plane1 = Line((boundary[0][0]+0.1, boundary[0][1]),(boundary[0][0]+0.1, boundary[1][1]), 1)
plane2 = Line((boundary[0][0]+0.2, boundary[0][1]),(boundary[0][0]+0.2, boundary[1][1]), 1)
plane3 = Line((boundary[0][0]+0.3, boundary[0][1]),(boundary[0][0]+0.3, boundary[1][1]), 1)
plane4 = Line((boundary[0][0]+0.4, boundary[0][1]),(boundary[0][0]+0.4, boundary[1][1]), 1)

# define sympy varaibles to parametize domain curves
x, y, t_symbol = Symbol('x'), Symbol('y'), Symbol('t')
time_range = {t_symbol: (0, 10)}


class VKVSTrain(TrainDomain):
  def __init__(self, **config):
    super(VKVSTrain, self).__init__()

    # left wall inlet
    leftWall = rec.boundary_bc(outvar_sympy={'u': vel, 'v': 0},
                          batch_size_per_area=5000,
                          lambda_sympy={'lambda_u': 1.0 - ((2 * Abs(y)) / height),  # weight edges to be zero
                                         'lambda_v': 1.0},
                          criteria=Eq(x, -width / 2),
                          param_ranges=time_range)
    self.add(leftWall, name="leftWall")

    """
    noSlipBC = geo.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                 batch_size_per_area=10000,
                                 criteria=not(Eq(x, width/2) or Eq(x, -width/2)))
    self.add(noSlipBC, name="NoSlip")
    """

    # no slip top wall
    topWall = rec.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                 batch_size_per_area=5000,
                                 criteria=Eq(y, height / 2),
                                 param_ranges=time_range)
    self.add(topWall, name="topWallNoSlip")

    # no slip bottom wall
    bottomWall = rec.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                 batch_size_per_area=5000,
                                 criteria=Eq(y, -height / 2),
                                 param_ranges=time_range)
    self.add(bottomWall, name="bottomWallNoSlip")

    # circle no slip
    circleBC = circle.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                 batch_size_per_area=5000,
                                 param_ranges=time_range)
    self.add(circleBC, name="circleNoSlip")

    # right wall outlet 0 pressure
    rightWall = rec.boundary_bc(outvar_sympy={'p' : 0},
                          batch_size_per_area=5000,
                          criteria=Eq(x, width / 2),
                          param_ranges=time_range)
    self.add(rightWall, name="rightWall")

    # interior
    interior = geo.interior_bc(outvar_sympy={'continuity': 0, 'momentum_x': 0, 'momentum_y': 0},
                               bounds={x: (-width / 2, width / 2),
                                       y: (-height / 2, height / 2)},
                               lambda_sympy={'lambda_continuity': geo.sdf,
                                             'lambda_momentum_x': geo.sdf,
                                             'lambda_momentum_y': geo.sdf},
                               #criteria=(sqrt((x - circle_pos[0])**2 + (y - circle_pos[1])**2) > radius) ,
                               #criteria=((x - circle_pos[0])**2 + (y - circle_pos[1])**2) > radius**2,
                               batch_size_per_area=200000,
                               param_ranges=time_range)
    self.add(interior, name="Interior")

    """
    # Integral Continuity Lines/Planes
    plane1Cont = plane1.boundary_bc(outvar_sympy={'integral_continuity': 1},
                                    batch_size_per_area=64,
                                    lambda_sympy={'lambda_integral_continuity': 0.1})
    plane2Cont = plane2.boundary_bc(outvar_sympy={'integral_continuity': 1},
                                    batch_size_per_area=64,
                                    lambda_sympy={'lambda_integral_continuity': 0.1})
    plane3Cont = plane3.boundary_bc(outvar_sympy={'integral_continuity': 1},
                                    batch_size_per_area=64,
                                    lambda_sympy={'lambda_integral_continuity': 0.1})
    plane4Cont = plane4.boundary_bc(outvar_sympy={'integral_continuity': 1},
                                    batch_size_per_area=64,
                                    lambda_sympy={'lambda_integral_continuity': 0.1})
    outletCont = rec.boundary_bc(outvar_sympy={'integral_continuity': 1},
                                    batch_size_per_area=64,
                                    lambda_sympy={'lambda_integral_continuity': 0.1},
                                    criteria=Eq(x, width / 2))

    self.add(plane1Cont, name="IntegralContinuity1")
    self.add(plane2Cont, name="IntegralContinuity2")
    self.add(plane3Cont, name="IntegralContinuity3")
    self.add(plane4Cont, name="IntegralContinuity4")
    self.add(outletCont, name="IntegralContinuity5")
    

    # interior
    interior = geo.interior_bc(outvar_sympy={'continuity': 0, 'momentum_x': 0, 'momentum_y': 0},
                               bounds={x: (-width / 2, width / 2),
                                       y: (-height / 2, height / 2)},
                               lambda_sympy={'lambda_continuity': geo.sdf,
                                             'lambda_momentum_x': geo.sdf,
                                             'lambda_momentum_y': geo.sdf},
                               #criteria=(sqrt((x - circle_pos[0])**2 + (y - circle_pos[1])**2) > radius) ,
                               criteria=((x - circle_pos[0])**2 + (y - circle_pos[1])**2) > radius**2,
                               batch_size_per_area=400000)
    self.add(interior, name="Interior")
    """


class VKVSVal(ValidationDomain):
  def __init__(self, **config):
    super(VKVSVal, self).__init__()
    #val = Validation.from_numpy(openfoam_invar_numpy, openfoam_outvar_numpy)
    #self.add(val, name='Val')


class VKVSInference(InferenceDomain):
  def __init__(self,**config):
    super(VKVSInference,self).__init__()
    #save entire domain
    interior = Inference(geo.sample_interior(1e5, bounds={x: (-width/2, width/2), y: (-height/2, height/2)}), ['u','v','p'])
    self.add(interior, name="Inference")


class VKVSMonitor(MonitorDomain):
  def __init__(self, **config):
    super(VKVSMonitor, self).__init__()
    # metric for mass imbalance, momentum imbalance and peak velocity magnitude
    #global_monitor = Monitor(geo.sample_interior(400000, bounds={x: (-width/2, width/2), y: (-height/2, height/2)}),
    #                         {'mass_imbalance': lambda var: tf.reduce_sum(var['area']*tf.abs(var['continuity'])),
    #                          'momentum_imbalance': lambda var: tf.reduce_sum(var['area']*tf.abs(var['momentum_x'])+tf.abs(var['momentum_x'])),
    #                         })
    #self.add(global_monitor, 'GlobalMonitor')

    force = Monitor(circle.sample_boundary(100),
                    {'force_x': lambda var: tf.reduce_sum(var['normal_x']*var['area']*var['p']),
                    'force_y': lambda var: tf.reduce_sum(var['normal_y']*var['area']*var['p'])})
    self.add(force, 'Force')


class VKVSSolver(Solver):
  train_domain = VKVSTrain
  #val_domain = VKVSVal
  inference_domain = VKVSInference
  #monitor_domain = VKVSMonitor

  def __init__(self, **config):
    super(VKVSSolver, self).__init__(**config)
    self.equations = (NavierStokes(nu=viscosity, rho=1.0, dim=2, time=True).make_node())
                      #+ IntegralContinuity().make_node())
    flow_net = self.arch.make_node(name='flow_net',
                                   inputs=['x', 'y', 't'],
                                   outputs=['u', 'v', 'p'])
    self.nets = [flow_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'layer_size': 256,
        'network_dir': './network_checkpoint_vkvs_re' + str(re),
        #'save_filetypes': 'csv,vtk,np',
        'decay_steps': 4000,
        'max_steps': 400000,
        'rec_results_cpu': True # New untested...
    })


if __name__ == '__main__':
  ctr = ModulusController(VKVSSolver)
  ctr.run()
