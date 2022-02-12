import glob
import numpy as np
import matplotlib.pyplot as plt
from modulus.csv_utils.csv_rw import csv_to_dict

# get list of files
files = glob.glob('./network_checkpoint_test_re_100/**/inference_domain/results/Inference_*.npz')
files.sort()
time_points = []
tke_points = []

# read each file and store tke
for i, f in enumerate(files):
  # read file
  predicted_data = np.load(f, allow_pickle=True)['arr_0'].item()

  if float(predicted_data['shifted_t'][0,0]) < 10.0:
    # store time
    time_points.append(float(predicted_data['shifted_t'][0,0]))
    
    # compute tke and store
    tke = np.mean(predicted_data['u']**2/2 + predicted_data['v']**2/2 + predicted_data['w']**2/2)
    tke_points.append(tke)
tke_points = np.array(tke_points)
tke_points = tke_points/np.max(tke_points)


# load validation tke data
#validation_tke_128 = csv_to_dict('validation_tke/tke_mean_Re500_N128.csv')
#validation_tke_256 = csv_to_dict('validation_tke/tke_mean_Re500_N256.csv')

"""
# plot turbulent kinetic energy decay
plt.plot(time_points, tke_points, label='Modulus')
plt.plot(validation_tke_128['Time'][:,0], validation_tke_128['TKE_mean'][:,0], label='Spectral Solver (grid res: 128)')
plt.plot(validation_tke_256['Time'][:,0], validation_tke_256['TKE_mean'][:,0], label='Spectral Solver (grid res: 256)')
plt.legend()
plt.title('TKE')
plt.ylabel('TKE')
plt.xlabel('time')
plt.savefig('tke_plot.png')
"""