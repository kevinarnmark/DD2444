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

# params for domain
height = 0.1
width = 0.5
radius = 0.01
circle_pos = (-width/4, 0)
vel = 1.0
boundary = ((-width / 2, -height / 2), (width / 2, height / 2))

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
x, y = Symbol('x'), Symbol('y')

class KarmanTrain(TrainDomain):
  def __init__(self, **config):
    super(KarmanTrain, self).__init__()

    #top wall
    #topWall = geo.boundary_bc(outvar_sympy={'u': vel, 'v': 0},
    #                          batch_size_per_area=10000,
    #                          lambda_sympy={'lambda_u': 1.0 - 20 * Abs(x),  # weight edges to be zero
    #                                        'lambda_v': 1.0},
    #                          criteria=Eq(y, height / 2))
    #self.add(topWall, name="TopWall")

    # left wall inlet
    leftWall = rec.boundary_bc(outvar_sympy={'u': vel, 'v': 0},
                          batch_size_per_area=10000,
                          lambda_sympy={'lambda_u': 1.0 - 20 * Abs(y),  # weight edges to be zero
                                         'lambda_v': 1.0},
                          criteria=Eq(x, -width / 2))
    self.add(leftWall, name="leftWall")
    """
    noSlipBC = geo.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                 batch_size_per_area=10000,
                                 criteria=not(Eq(x, width/2) or Eq(x, -width/2)))
    self.add(noSlipBC, name="NoSlip")
    """

    # no slip top wall
    topWall = rec.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                 batch_size_per_area=10000,
                                 criteria=Eq(y, height / 2))
    self.add(topWall, name="topWallNoSlip")

    # no slip bottom wall
    bottomWall = rec.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                 batch_size_per_area=10000,
                                 criteria=Eq(y, -height / 2))
    self.add(bottomWall, name="bottomWallNoSlip")

    # circle no slip
    circleBC = circle.boundary_bc(outvar_sympy={'u': 0, 'v': 0},
                                 batch_size_per_area=10000,)
    self.add(circleBC, name="circleNoSlip")

    # right wall outlet 0 pressure
    rightWall = rec.boundary_bc(outvar_sympy={'p' : 0},
                          batch_size_per_area=10000,
                          criteria=Eq(x, width / 2))
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
                               batch_size_per_area=400000)
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


class KarmanVal(ValidationDomain):
  def __init__(self, **config):
    super(KarmanVal, self).__init__()
    #val = Validation.from_numpy(openfoam_invar_numpy, openfoam_outvar_numpy)
    #self.add(val, name='Val')


class KarmanInference(InferenceDomain):
  def __init__(self,**config):
    super(KarmanInference,self).__init__()
    #save entire domain
    interior = Inference(geo.sample_interior(1e06, bounds={x: (-width/2, width/2), y: (-height/2, height/2)}), ['u','v','p'])
    self.add(interior, name="Inference")


class KarmanMonitor(MonitorDomain):
  def __init__(self, **config):
    super(KarmanMonitor, self).__init__()
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


class KarmanSolver(Solver):
  train_domain = KarmanTrain
  #val_domain = KarmanVal
  inference_domain = KarmanInference
  monitor_domain = KarmanMonitor

  def __init__(self, **config):
    super(KarmanSolver, self).__init__(**config)
    self.equations = (NavierStokes(nu=0.001, rho=1.0, dim=2, time=False).make_node())
                      #+ IntegralContinuity().make_node())
    flow_net = self.arch.make_node(name='flow_net',
                                   inputs=['x', 'y'],
                                   outputs=['u', 'v', 'p'])
    self.nets = [flow_net]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'network_dir': './network_checkpoint_von_karman',
        #'save_filetypes': 'csv,vtk,np',
        'decay_steps': 4000,
        'max_steps': 400000
    })


if __name__ == '__main__':
  ctr = ModulusController(KarmanSolver)
  ctr.run()
