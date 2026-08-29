import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools
G = 1
rho_crit = 1.0  # Critical maximum density threshold in QLC
#other constants are all over the place, just assume 0
def lqc_dynamic_system(t, state):
    """state vector
    state[0, 1, 2] are for each separate scaling factor
    state[3] is for hubble parameter"""
    a_x = state[0]
    a_y = state[1]
    a_z = state[2]
    H = state[3]
    
    #avoid div0 errors
    if np.isclose(a_x or a_y or a_z ,0, atol=1e-5):
        return [0.0, 0.0]
        
    #conservation of momentum, must be >1
    p_phi = 15.0 
    V = a_x * a_y * a_z
    # Energy density of scalar field: \rho = p_\phi^2 / (2 * a^6)
    rho = (p_phi**2) / (2.0 * V**2)
    
    #scale factor derivative: \dot{a} = a * H, from Friedmann
    dax_dt = a_x * H
    day_dt = a_y * H
    daz_dt = a_z * H
    
    # modified accelerated friedmann equation \dot{H} = -4 * \pi * G * \rho * (1 - 2\rho/\rho_crit)
    # rho > 0.5 rho_crit was reccomended by someone online. If someone could explain it that would be nice
    dH_dt = -4.0 * np.pi * G * rho * (1.0 - (2.0 * rho / rho_crit))
    
    return [dax_dt, day_dt, daz_dt, dH_dt]

x_scale = np.linspace(11, 21, 10)
y_scale = np.linspace(11, 21, 10)
z_scale = np.linspace(11, 21, 10)

p_phi_val = 15  # same as p_phi
time_span = (0, 10)  #next time step
t_eval = np.linspace(time_span[0], time_span[1], 1000)

results = []

for initial_a_x, initial_a_y, initial_a_z in itertools.product(
    x_scale, y_scale, z_scale
):
  V = initial_a_x * initial_a_y * initial_a_z
  initial_rho = (p_phi_val**2) / (2.0 * V**2)

  #start h as negative
  initial_H = -np.sqrt((8.0 * np.pi * G / 3.0)* initial_rho* (1.0 - (initial_rho / rho_crit))  )
  initial_state = [initial_a_x, initial_a_y, initial_a_z, initial_H]

  #RK45 is runge kutta 4/5th order, DOP853 is dormand prince 8th order; can change im pretty sure it still works
  solution = solve_ivp(
      lqc_dynamic_system, time_span, initial_state, t_eval=t_eval, method='RK45',
  )

  #extract solutions of each equation
  t_steps = solution.t
  ax_data = solution.y[0]
  ay_data = solution.y[1]
  az_data = solution.y[2]
  H_data = solution.y[3]
  rho_data = (p_phi_val**2) / (2.0 * (ax_data * ay_data*az_data)**2)

  # Store results
  results.append(
      {
          'params': (initial_a_x, initial_a_y, initial_a_z),
          't': t_steps,'ax': ax_data, 'ay': ay_data, 'az': az_data, 'H': H_data, 'rho': rho_data,
      }
  )

print(results) #i forgot how to print a graph or collapse a 4d graph into 2d with heat sigs