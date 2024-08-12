import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#import lecroy
import h5py
import os
import sys
from scipy.signal import periodogram, welch
from scipy.fft import rfft, irfft, rfftfreq
from scipy.optimize import curve_fit, least_squares

def lin2db(lin):
    return 10 * np.log10(lin)

def db2lin(db):
    return 10**(db/10)

def read_file_to_hdf5(target_folder,target_files):
    sq_5mhz_files = [os.path.join(target_folder, f) for f in sorted(target_files) if '5MHz' in f and '_S' in f]
    asq_5mhz_files = [os.path.join(target_folder, f) for f in sorted(target_files) if '5MHz' in f and '_A' in f]
    elec_5mhz_file = os.path.join(target_folder, '5MHz_elec.csv')
    vac_5mhz_file = os.path.join(target_folder, '5MHz_vac.csv')

    electronic = np.genfromtxt(elec_5mhz_file, delimiter=',', skip_header=45)
    vacuum = np.genfromtxt(vac_5mhz_file, delimiter=',', skip_header=45)
    squeezing = np.array([np.genfromtxt(f, delimiter=',', skip_header=45) for f in sq_5mhz_files])
    antisqueezing = np.array([np.genfromtxt(f, delimiter=',', skip_header=45) for f in asq_5mhz_files])

    powers = []
    for p in sq_5mhz_files:
        powers.append(p.split("_")[1])
    powers = np.array(powers).astype(float)

    meta_elec = np.genfromtxt(elec_5mhz_file, delimiter=',', skip_header=2, max_rows=42, dtype='S')

    with h5py.File('squeezing_data.hdf5', 'w') as file:
        dset_spec = file.create_group('spectrum')
        dset_power = dset_spec.create_dataset('powers', data=powers)
        dset_elec = dset_spec.create_dataset('electronic', data=electronic)
        dset_vac = dset_spec.create_dataset('vacuum', data=vacuum)
        dset_sqz = dset_spec.create_dataset('squeezing', data=squeezing)
        dset_asqz = dset_spec.create_dataset('antisqueezing', data=antisqueezing)

        for key, val in meta_elec:
            dset_spec.attrs[key] = val

        vacuum_corr_lin = db2lin(vacuum) - db2lin(electronic)
        squeezing_norm = lin2db((db2lin(squeezing) - db2lin(electronic)) / vacuum_corr_lin)
        antisqueezing_norm = lin2db((db2lin(antisqueezing) - db2lin(electronic)) / vacuum_corr_lin)

    return powers,electronic,vacuum,squeezing_norm[:,:,1],antisqueezing_norm[:,:,1]

if __name__ == '__main__':
    jens_folder = 'dataset/jens'
    jens_files = os.listdir(jens_folder)
    [powers,electronic,vacuum,squeezing,antisqueezing] = read_file_to_hdf5(jens_folder,jens_files)
