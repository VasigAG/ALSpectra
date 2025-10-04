# all the packlages to be imported 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
import scipy

# creating a pandas dataframe to store the patients data
patient_data = pd.read_excel("task1/sand_task_1.xlsx")

# contains ID, Age, Sex, Class
# ALSFRS-R score, an integer value from 0 to 4 for ALS patients, or a value of 5 for the healthy subjects.
# 4 is good functioning, 1 is really bad
# preprocessing signal class

patient_data = pd.read_excel("task1/sand_task_1.xlsx")

class signal:

    patient_data = pd.read_excel("task1/sand_task_1.xlsx")

    def __init__(self,id,sound):
        self.a = wavfile.read(f"task1/training/phonationA/{id}_phonation{sound}.wav") # (int:sample_rate, np.array:waveform)
        (self.sample_rate,self.wav) = self.a
        self.clas = int(patient_data["Class"][np.where(patient_data["ID"] == id)[0]].iloc[0])
        self.age = int(patient_data["Age"][np.where(patient_data["ID"] == id)[0]].iloc[0])
        self.sex = 0 if (patient_data["Sex"][np.where(patient_data["ID"] == id)[0]].iloc[0])=="M" else "F"

    def plot_fft(self):
        # sample_rate,waveform = self.a
        fft_ = scipy.fft.fft(self.wav)
        len_ = len(self.wav)
        fft_mag = abs(fft_)
        freq_buckets = np.linspace(0,self.sample_rate,len(fft_))
        plt.plot(freq_buckets,fft_mag)
    
    def get_fft(self):
        fft_ = scipy.fft.fft(self.wav)
        len_ = len(self.wav)
        fft_mag = abs(fft_)
        freq_buckets = np.linspace(0,self.sample_rate,len_)
        return fft_

    def __len__(self):
        return len(self.wav)
    

