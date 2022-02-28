from fenics import *
import numpy as np

# Create mesh and define function space
mesh = UnitSquareMesh(128, 128)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
#u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
u_D = Constant(0.0)
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, 0.0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
#f = Expression("(1/2)*(sin(pi*x[0])*sin(pi*x[1]) - 2*sin(2*pi*x[0])*sin(2*pi*x[1]) + 3*sin(3*pi*x[0])*sin(3*pi*x[1]) - 4*sin(4*pi*x[0])*sin(4*pi*x[1]))", pi=np.pi, degree=2)
f = Expression("0.5*(sin(pi*x[0])*sin(pi*x[1]) - 2*sin(2*pi*x[0])*sin(2*pi*x[1]) + 3*sin(3*pi*x[0])*sin(3*pi*x[1]) - 4*sin(4*pi*x[0])*sin(4*pi*x[1]))", degree=1, pi=np.pi)
#f = Constant(10)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution

u = Function(V)

solve(a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)

# Save solution to file in VTK format
vtkfile = File('poisson/solution2.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
#interactive()