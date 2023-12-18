# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=6):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def read_file(file):
    data = pd.read_csv(file)
    return data

def compute_DE(signal):
	variance = np.var(signal,ddof=1)
	return math.log(2*math.pi*math.e*variance)/2

def decompose(file):
    data = file
    shape = data.shape
    frequency = 250

    decomposed_de = np.empty([0,5,8])

    base_DE = np.empty([0,250])


    temp_base_DE = np.empty([0])
    temp_base_delta_DE = np.empty([0])
    temp_base_theta_DE = np.empty([0])
	temp_base_alpha_DE = np.empty([0])
	temp_base_beta_DE = np.empty([0])
	temp_base_gamma_DE = np.empty([0])

	temp_de = np.empty([0,8])

	for channel in range(128):
		trial_signal = data.loc[channel,500:]
		base_signal = data.loc[channel,:500]
		
		base_delta = butter_bandpass_filter(base_signal, 0.5, 4, frequency, order=6)
		base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=6)
		base_alpha = butter_bandpass_filter(base_signal, 8,14, frequency, order=6)
		base_beta = butter_bandpass_filter(base_signal,14,31, frequency, order=6)
		base_gamma = butter_bandpass_filter(base_signal,31,50, frequency, order=6)

		base_delta_DE = (compute_DE(base_delta[:250])+compute_DE(base_delta[250:500]))/2
		base_theta_DE = (compute_DE(base_theta[:250])+compute_DE(base_theta[250:500]))/2
		base_alpha_DE =(compute_DE(base_alpha[:250])+compute_DE(base_alpha[250:500]))/2
		base_beta_DE =(compute_DE(base_beta[:250])+compute_DE(base_beta[250:500]))/2
		base_gamma_DE =(compute_DE(base_gamma[:250])+compute_DE(base_gamma[250:500]))/2

		temp_base_delta_DE = np.append(temp_base_delta_DE,base_delta_DE)
		temp_base_theta_DE = np.append(temp_base_theta_DE,base_theta_DE)
		temp_base_gamma_DE = np.append(temp_base_gamma_DE,base_gamma_DE)
		temp_base_beta_DE = np.append(temp_base_beta_DE,base_beta_DE)
		temp_base_alpha_DE = np.append(temp_base_alpha_DE,base_alpha_DE)

		delta = butter_bandpass_filter(trial_signal, 0.5, 4, frequency, order=6)
		theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=6)
		alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=6)
		beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=6)
		gamma = butter_bandpass_filter(trial_signal, 31, 50, frequency, order=6)

		DE_delta = np.zeros(shape=[0],dtype = float)
		DE_theta = np.zeros(shape=[0],dtype = float)
		DE_alpha = np.zeros(shape=[0],dtype = float)
		DE_beta =  np.zeros(shape=[0],dtype = float)
		DE_gamma = np.zeros(shape=[0],dtype = float)
        
		for index in range(8):
				DE_delta =np.append(DE_delta,compute_DE(delta[index*frequency:(index+1)*frequency]))            
				DE_theta =np.append(DE_theta,compute_DE(theta[index*frequency:(index+1)*frequency]))
				DE_alpha =np.append(DE_alpha,compute_DE(alpha[index*frequency:(index+1)*frequency]))
				DE_beta =np.append(DE_beta,compute_DE(beta[index*frequency:(index+1)*frequency]))
				DE_gamma =np.append(DE_gamma,compute_DE(gamma[index*frequency:(index+1)*frequency])) 
				#DE_all =np.vstack([DE_delta,DE_theta,DE_alpha,DE_beta,DE_gamma]) 

		temp_de = np.vstack([temp_de,DE_delta])				
		temp_de = np.vstack([temp_de,DE_theta])	
		temp_de = np.vstack([temp_de,DE_alpha])
		temp_de = np.vstack([temp_de,DE_beta])
		temp_de = np.vstack([temp_de,DE_gamma])
		temp_trial_de = temp_de.reshape(-1,5,8)
		decomposed_de = np.vstack([decomposed_de,temp_trial_de])
        
		temp_base_DE = np.append(temp_base_delta_DE,temp_base_theta_DE)
		temp_base_DE = np.append(temp_base_DE,temp_base_alpha_DE)
		temp_base_DE = np.append(temp_base_DE,temp_base_beta_DE)
		temp_base_DE = np.append(temp_base_DE,temp_base_gamma_DE)
       
		temp_base_DEE = temp_base_DE.reshape(5,-1)
		original_DE = temp_base_DEE.transpose()
		original_DE=original_DE.reshape(-1,5,1)
		original_total_DE = [original_DE,temp_trial_de]
		total_DE = np.concatenate(original_total_DE,axis=2)

	print("original_DE shape:",original_DE.shape)
	print("total_DE shape:",total_DE.shape)
	finish_data= total_DE
	return original_DE,total_DE,finish_data
   

if __name__ == '__main__':
    dataset_dir = "D:\EEG_Segments"
    result_dir = "D:\EEG_DE"
    if os.path.isdir(result_dir)==False:
       os.makedirs(result_dir)

    for file in os.listdir(dataset_dir):
       print("processing: ",file,"......")
       file_path = pd.read_csv(dataset_dir+"/"+file,header=None)
       original_DE,total_DE,finish_data = decompose(file_path)
       np.save("E:\EEG_DE"+"/"+"DE_"+file[:-5], arr=finish_data)
