import numpy as np
import os
current = os.getcwd()
sim_file = "initial_samples.npz"
data = np.load(current + '/' + sim_file)

X = data['arr_0']
Y = data['arr_1']

Y[:,0] = -Y[:,0]



    
try:     
    np.savez(current + '/' + sim_file, X, Y)
    print('Data saved in file: ', sim_file, ' directory ', str(current), ' X ', X, ' Y ', Y)
except Exception as e:
    print(e)
    print('Data saving failed to initial_samples file.')