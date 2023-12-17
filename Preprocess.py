# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
from scipy import signal
from sklearn import preprocessing


read_path = ""

output_path = ""


def get_file_path():
    read_path = r"D:\EEG128"
    output_path = r"E:\EEGpreprocess"
    return read_path,output_path


def deal_files():

    files = os.listdir(read_path)
    for file_name in files:
        dfdata = pd.read_csv(read_path+"/"+file_name)
        finish_dfdata = get_deal_file(dfdata)
        finish_dfdata.to_csv(output_path + "/" + "pre" + file_name, index=False)

    

def get_deal_file(dfdata):
    datapre=dfdata.drop(dfdata.columns[[0]], axis=1)
    dataready=datapre.drop(datapre.index[[128]])
    rawdata=np.array(dataready)
    
    #cut data
    a=len(rawdata[0])
    cutdata=rawdata[:,7500:(a-7500)]
    
    '''butterworth'''
    def butterBandPassFilter(lowcut, highcut, samplerate, order):
        semiSampleRate = samplerate*0.5 
        low = lowcut / semiSampleRate
        high = highcut / semiSampleRate
        b,a = signal.butter(order,[low,high],btype='bandpass')
        return b,a
    def butterBandStopFilter(lowcut, highcut, samplerate, order):
        semiSampleRate = samplerate*0.5
        low = lowcut / semiSampleRate
        high = highcut / semiSampleRate
        b,a = signal.butter(order,[low,high],btype='bandstop')
        return b,a

    iSampleRate = 250  
    b,a = butterBandPassFilter(0.5,50,iSampleRate,order=6)
    x1 = signal.lfilter(b,a,cutdata)  


    b,a = butterBandStopFilter(48,52,iSampleRate,order=6)
    databutter = signal.lfilter(b,a,x1)
    
    
    dataready = preprocessing.scale(databutter,axis=0)
    
    finish_dfdata= pd.DataFrame(dataready)
    return finish_dfdata



if __name__=="__main__":

    read_path,output_path = get_file_path()
    deal_files()