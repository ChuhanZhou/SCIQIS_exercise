import typing as tp
import numpy as np
import h5py
from scipy.fft import rfft, irfft, rfftfreq
from scipy.optimize import curve_fit, least_squares
import os
import process_data
import matplotlib.pyplot as plt

Vsq = lambda P, Pth, eta, frel: 1 - 4*eta*np.sqrt(P/Pth) / ((1+np.sqrt(P/Pth))**2 + frel**2)
Vasq = lambda P, Pth, eta, frel: 1 + 4*eta*np.sqrt(P/Pth) / ((1-np.sqrt(P/Pth))**2 + frel**2)
Vrot = lambda P, Pth, eta, frel, theta: np.cos(theta*np.pi/180)**2 * Vsq(P, Pth, eta, frel) + np.sin(theta*np.pi/180)**2 * Vasq(P, Pth, eta, frel)

def Vrotfluct(P, Pth, eta, frel, theta, sigma_x=0, sigma_p=None):
    if sigma_p is None:
        sigma_p = sigma_x
    
    Vx = .5 * Vsq(P, Pth, eta, frel) * (1 + np.cos(2 * theta * np.pi/180) * np.exp(-2 * (sigma_x * np.pi/180)**2))
    Vp = .5 * Vasq(P, Pth, eta, frel) * (1 - np.cos(2 * theta * np.pi/180) * np.exp(-2 * (sigma_p * np.pi/180)**2))

    return Vx + Vp

def lin2db(lin):
    return 10 * np.log10(lin)

def db2lin(db):
    return 10**(db/10)

def calc_noise_var(list_traces:np.ndarray):
    return np.mean(list_traces,axis=1)

def res_model(params, P, sq_spec, asq_spec):
    Pth, eta, norm_f_bw, theta_sq, theta_asq, sigma = params
    model_sq = lin2db(Vrotfluct(P, Pth, eta, norm_f_bw, theta_sq, sigma))
    model_asq = lin2db(Vrotfluct(P, Pth, eta, norm_f_bw, theta_asq, sigma))
    residuals = np.concatenate([sq_spec-model_sq, asq_spec-model_asq])
    return residuals


if __name__=="__main__":
    jens_folder = 'dataset/jens'
    jens_files = os.listdir(jens_folder)
    [powers, electronic, vacuum, squeezing, antisqueezing] = process_data.read_file_to_hdf5(jens_folder, jens_files)
    powers=powers
    spectra_asq=antisqueezing
    spectra_sq=squeezing
    freqs=None#[]
    vars_asq=calc_noise_var(spectra_asq)
    vars_sq=calc_noise_var(spectra_sq)





    res = least_squares(res_model, [1.5,.8,0.2, 0, 90, 0],
                    bounds=([0,.5,0,-10,70,-10], [20,1,0.4,10,110,10]),
                    args=[powers, vars_sq, vars_asq])
    Pth, eta, norm_f_bw, theta_sq, theta_asq, sigma = res.x
    model_sq = lin2db(Vrotfluct(np.linspace(0, 5, 1000), Pth, eta, norm_f_bw, theta_sq, sigma))
    model_asq = lin2db(Vrotfluct(np.linspace(0, 5, 1000), Pth, eta, norm_f_bw, theta_asq, sigma))
    plt.plot(powers, vars_asq, "s",label="Anti-Squeezing")
    plt.plot(powers, vars_sq, "s",label= "Squeezing")
    plt.plot(np.linspace(0, 5, 1000),model_asq,":",label= "Anti-Squeezing w/o Phase Noise")
    plt.plot(np.linspace(0, 5, 1000),model_sq, ":",label= "Squeezing w/o Phase Noise")
    plt.ylabel("Relative Noise Power [dB]")
    plt.xlabel("Pump Power [mW]")
    plt.legend()
    plt.show()