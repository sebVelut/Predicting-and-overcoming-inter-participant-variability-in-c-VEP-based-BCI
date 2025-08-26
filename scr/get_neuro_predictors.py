import numpy as np
import pandas as pd
import mne
from scipy import stats
from skimage.measure import block_reduce
import math
from scipy import stats
from utils.STLDataLoader import STLDataLoader
from scipy.integrate import simpson
from utils.Green_files.green.wavelet_layers import WaveletConv
import torch




##################### BP ###########################
participants = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24']
# participants = ['P1','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24']
dl = STLDataLoader('D:/s.velut/Documents/These/Protheus_PHD/Data/Dry_Ricker/', 1, 40, 0.35, 500, 60, "time_sample", participants, 5)

n_subjects = len(participants)
raw_data = dl.load_data()
# epochs,elabels = dl.get_mne_epochs(raw_data)
X_parent, Y, domains, codes, labels_code = dl.get_epochs(raw_data)

datas = X_parent[:,:9250]
Y = Y[:,:9250]
channel = 4

bt_amp = np.zeros(n_subjects)
bt_pic = np.zeros(n_subjects)
for i in range(n_subjects):
    bt_amp[i] = np.max(np.abs(np.max(datas[i][Y[i]==1,4,:],axis=0) - np.min(datas[i][Y[i]==1,4,:],axis=0)),axis=0)
    bt_pic[i] = np.argmax(np.abs(np.mean(datas[i][Y[i]==1,4,:],axis=0)))


bt_SNR = []
for i in range(24):
    spectrum1 = mne.EpochsArray(datas[i][Y[i]==1],raw_data[i].info,verbose=False).compute_psd()
    data1 = spectrum1.get_data(fmin=8, fmax=13)
    fq_res1 = spectrum1.freqs[1] - spectrum1.freqs[0]
    bp1 = np.average(simpson(data1, dx=fq_res1, axis=-1),axis=0)  # (10, 3)

    spectruma = mne.EpochsArray(datas[i],raw_data[i].info,verbose=False).compute_psd()
    dataa = spectruma.get_data(fmin=8, fmax=13)
    fq_resa = spectruma.freqs[1] - spectruma.freqs[0]
    bpa = np.mean(simpson(dataa, dx=fq_resa, axis=-1),axis=0)  # (10, 3)
    bt_SNR.append(bp1/bpa)

bt_spectrums = []
for i in range(24):
    bt_spectrums.append(mne.EpochsArray(datas[i][Y[i]==1],raw_data[i].info,verbose=False).compute_psd(method='welch',fmin=1.0, fmax=45.0, tmax=2.0, n_jobs=None,verbose=False,n_fft=175))
bt_alphas = np.zeros((n_subjects,datas[0][Y[i]==1].shape[0]))
bt_betas = np.zeros((n_subjects,datas[0][Y[i]==1].shape[0]))
bt_deltas = np.zeros((n_subjects,datas[0][Y[i]==1].shape[0]))
bt_thetas = np.zeros((n_subjects,datas[0][Y[i]==1].shape[0]))
for i in range(n_subjects):
    bt_alphas[i] = np.mean((mne.EpochsArray(datas[i][Y[i]==1],raw_data[i].info,verbose=False).compute_psd(method='welch',fmin=8.0, fmax=13.0, tmax=2.0, n_jobs=None,verbose=False,n_fft=175)).get_data(),axis=1).max(axis=1)/np.mean(bt_spectrums[i])
    bt_betas[i] = np.mean((mne.EpochsArray(datas[i][Y[i]==1],raw_data[i].info,verbose=False).compute_psd(method='welch',fmin=13.0, fmax=35.0, tmax=2.0, n_jobs=None,verbose=False,n_fft=175)).get_data(),axis=1).max(axis=1)/np.mean(bt_spectrums[i])
    bt_deltas[i] = np.mean((mne.EpochsArray(datas[i][Y[i]==1],raw_data[i].info,verbose=False).compute_psd(method='welch',fmin=0.5, fmax=4.0, tmax=2.0, n_jobs=None,verbose=False,n_fft=175)).get_data(),axis=1).max(axis=1)/np.mean(bt_spectrums[i])
    bt_thetas[i] = np.mean((mne.EpochsArray(datas[i][Y[i]==1],raw_data[i].info,verbose=False).compute_psd(method='welch',fmin=4.0, fmax=8.0, tmax=2.0, n_jobs=None,verbose=False,n_fft=175)).get_data(),axis=1).max(axis=1)/np.mean(bt_spectrums[i])
    

bt_m = np.zeros(n_subjects)
bt_all_cm = np.zeros((n_subjects,math.ceil(datas[0][Y[0]==1].shape[0]/8),math.ceil(datas[0][Y[0]==1].shape[0]/8))) 
for i in range(n_subjects):
    corr = stats.spearmanr(datas[i][Y[i]==1,4,:],axis=1)
    arr_reduced = block_reduce(corr.statistic, block_size=(8,8), func=np.median, cval=np.median(corr))

    bt_m[i] = np.median(arr_reduced)
    bt_all_cm[i] = arr_reduced

####################### AP ###########################
    
files_path = "../Data/STL/ws0.35/time_sample/data/cal_2/"
datas = np.array([np.load(files_path+"full_solo_preprocess_data_P"+str(i)+".npy") for i in range(1,25) if i!=26])
datas[1,:,0,:] = datas[1,:,1,:]
datas[1,:,1,:] = datas[1,:,2,:]
datas[1,:,2,:] = datas[1,:,3,:]
n_subjects = datas.shape[0]
datas = datas[:,:9250]
Y = Y[:,:9250]
channel = 0

at_amp = np.zeros(n_subjects)
at_pic = np.zeros(n_subjects)
for i in range(n_subjects):
    at_amp[i] = np.max(np.mean(datas[i][Y[i]==1,0,:],axis=0)) - np.min(np.mean(datas[i][Y[i]==1,0,:],axis=0))
    at_pic[i] = np.argmax(np.abs(np.mean(datas[i][Y[i]==1,0,:],axis=0)))

at_SNR = []
for i in range(24):
    spectrum1 = mne.EpochsArray(datas[i][Y[i]==1][:,:3],mne.create_info(["001","002","003"],500,'eeg',False),verbose=False).compute_psd()
    data1 = spectrum1.get_data(fmin=8, fmax=13)
    fq_res1 = spectrum1.freqs[1] - spectrum1.freqs[0]
    bp1 = np.average(simpson(data1, dx=fq_res1, axis=-1),axis=0)  # (10, 3)

    spectruma = mne.EpochsArray(datas[i][:,:3],mne.create_info(["001","002","003"],500,'eeg',False),verbose=False).compute_psd()
    dataa = spectruma.get_data(fmin=8, fmax=13)
    fq_resa = spectruma.freqs[1] - spectruma.freqs[0]
    bpa = np.mean(simpson(dataa, dx=fq_resa, axis=-1),axis=0)  # (10, 3)

    at_SNR.append(bp1/bpa)

at_spectrums = []
for i in range(24):
    at_spectrums.append(mne.EpochsArray(datas[i][Y[i]==1][:,:1],mne.create_info(["001",],500,'eeg',False),verbose=False).compute_psd(method='welch',fmin=1.0, fmax=60.0, tmax=2.0, n_jobs=None,verbose=False,n_fft=175))
at_alphas = np.zeros((n_subjects,datas[0][Y[i]==1].shape[0]))
at_betas = np.zeros((n_subjects,datas[0][Y[i]==1].shape[0]))
at_deltas = np.zeros((n_subjects,datas[0][Y[i]==1].shape[0]))
at_thetas = np.zeros((n_subjects,datas[0][Y[i]==1].shape[0]))
for i in range(n_subjects):
    at_alphas[i] = np.mean((mne.EpochsArray(datas[i][Y[i]==1][:,:3],mne.create_info(["001","002","003"],500,'eeg',False),verbose=False).compute_psd(method='welch',fmin=8.0, fmax=13.0, tmax=2.0, n_jobs=None,verbose=False,n_fft=175)).get_data(),axis=1).max(axis=1)/np.mean(at_spectrums[i])
    at_betas[i] = np.mean((mne.EpochsArray(datas[i][Y[i]==1][:,:3],mne.create_info(["001","002","003"],500,'eeg',False),verbose=False).compute_psd(method='welch',fmin=13.0, fmax=35.0, tmax=2.0, n_jobs=None,verbose=False,n_fft=175)).get_data(),axis=1).max(axis=1)/np.mean(at_spectrums[i])
    at_deltas[i] = np.mean((mne.EpochsArray(datas[i][Y[i]==1][:,:3],mne.create_info(["001","002","003"],500,'eeg',False),verbose=False).compute_psd(method='welch',fmin=0.5, fmax=4.0, tmax=2.0, n_jobs=None,verbose=False,n_fft=175)).get_data(),axis=1).max(axis=1)/np.mean(at_spectrums[i])
    at_thetas[i] = np.mean((mne.EpochsArray(datas[i][Y[i]==1][:,:3],mne.create_info(["001","002","003"],500,'eeg',False),verbose=False).compute_psd(method='welch',fmin=4.0, fmax=8.0, tmax=2.0, n_jobs=None,verbose=False,n_fft=175)).get_data(),axis=1).max(axis=1)/np.mean(at_spectrums[i])

at_m = np.zeros(n_subjects)
at_all_cm = np.zeros((n_subjects,math.ceil(datas[0][Y[i]==1].shape[0]/8),math.ceil(datas[0][Y[i]==1].shape[0]/8)))
for i in range(n_subjects):
    corr = stats.spearmanr(datas[i][Y[i]==1][:,channel],axis=1)
    arr_reduced = block_reduce(corr.statistic, block_size=(8,8), func=np.median, cval=np.median(corr))
    at_m[i] = np.median(arr_reduced)
    at_all_cm[i] = arr_reduced


######################## BPWave ###########################
    
fwhm = np.load("../complements/param_wavelets_fwhm.npy")
foi = np.load("../complements/param_wavelets_foi.npy")
tt = np.load("../complements/param_wavelets_tt.npy")

participants = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24']
dl = STLDataLoader('D:/s.velut/Documents/These/Protheus_PHD/Data/Dry_Ricker/', 1, 45, 0.35, 500, 60, "time_sample", participants, 5)
raw_data = dl.load_data()
X_parent, Y, domains, codes, labels_code = dl.get_epochs(raw_data)
bt_datas = X_parent[:,:9250]
Y = Y[:,:9250]
channel = 4

bt_waves_real = np.zeros((bt_datas.shape[0],bt_datas[0][Y[0]==1].shape[0],foi.shape[1],bt_datas.shape[2],bt_datas.shape[3]))
bt_waves_imag = np.zeros((bt_datas.shape[0],bt_datas[0][Y[0]==1].shape[0],foi.shape[1],bt_datas.shape[2],bt_datas.shape[3]))
for i in range(n_subjects):

    wavelet = WaveletConv(
                kernel_width_s=0.35,
                sfreq=500,
                foi_init=foi[i],
                fwhm_init=fwhm[i],
                stride=1,
                dtype=torch.complex64,
                padding='same',
                scaling='oct'
            )
    bt_waves_real[i] = np.real(wavelet(torch.Tensor(bt_datas[i][Y[i]==1])).detach().numpy())
    bt_waves_imag[i] = np.imag(wavelet(torch.Tensor(bt_datas[i][Y[i]==1])).detach().numpy())

bt_waves_real_amp = np.zeros(n_subjects)
bt_waves_real_pic = np.zeros(n_subjects)
bt_waves_imag_amp = np.zeros(n_subjects)
bt_waves_imag_pic = np.zeros(n_subjects)
for i in range(n_subjects):
    bt_waves_real_amp[i] = np.max(np.mean(np.mean(bt_waves_real[i],axis=(0,2)),axis=0)) - np.min(np.mean(np.mean(bt_waves_real[i],axis=(0,2)),axis=0))
    bt_waves_real_pic[i] = np.argmax(np.abs(np.mean(np.mean(bt_waves_real[i],axis=0)[:,0],axis=0)))

    bt_waves_imag_amp[i] = np.max(np.mean(np.mean(bt_waves_imag[i],axis=(0,2)),axis=0)) - np.min(np.mean(np.mean(bt_waves_imag[i],axis=(0,2)),axis=0))
    bt_waves_imag_pic[i] = np.argmax(np.abs(np.mean(np.mean(bt_waves_imag[i],axis=0)[:,0],axis=0)))


bt_waves_real_m = np.zeros(n_subjects)
bt_all_cm_real = np.zeros((n_subjects,math.ceil(bt_datas[0][Y[i]==1].shape[0]/8),math.ceil(bt_datas[0][Y[i]==1].shape[0]/8)))
bt_waves_imag_m = np.zeros(n_subjects)
bt_all_cm_imag_imag = np.zeros((n_subjects,math.ceil(datas[0][Y[i]==1].shape[0]/8),math.ceil(datas[0][Y[i]==1].shape[0]/8)))
for i in range(n_subjects):
    corr = stats.spearmanr(np.mean(bt_waves_real[i],axis=2).reshape(-1,22*bt_waves_real.shape[-1]),axis=1)
    arr_reduced = block_reduce(corr.statistic, block_size=(8,8), func=np.median, cval=np.median(corr))
    bt_waves_real_m[i] = np.median(arr_reduced)
    bt_all_cm_real[i] = arr_reduced

    corr = stats.spearmanr(np.mean(bt_waves_imag[i],axis=2).reshape(-1,22*bt_waves_imag.shape[-1]),axis=1)
    arr_reduced = block_reduce(corr.statistic, block_size=(8,8), func=np.median, cval=np.median(corr))
    bt_waves_imag_m[i] = np.median(arr_reduced)
    bt_all_cm_imag_imag[i] = arr_reduced

########################### APWave ###########################
    
files_path = "D:/s.velut/Documents/These/Protheus_PHD/Data/STL/ws0.35/time_sample/data/cal_2/"
datas = np.array([np.load(files_path+"full_solo_preprocess_data_P"+str(i)+".npy") for i in range(1,25) if i!=26])
datas[1,:,0,:] = datas[1,:,1,:]
n_subjects = datas.shape[0]
datas = datas[:,:9250]
Y = Y[:,:9250]
channel = 0

waves_real = np.zeros((datas.shape[0],datas[0][Y[0]==1].shape[0],foi.shape[1],datas.shape[2],datas.shape[3]))
waves_imag = np.zeros((datas.shape[0],datas[0][Y[0]==1].shape[0],foi.shape[1],datas.shape[2],datas.shape[3]))
for i in range(n_subjects):
    wavelet = WaveletConv(
                kernel_width_s=0.35,
                sfreq=500,
                foi_init=foi[i],
                fwhm_init=fwhm[i],
                stride=1,
                dtype=torch.complex64,
                padding='same',
                scaling='oct'
            )
    waves_real[i] = np.real(wavelet(torch.Tensor(datas[i][Y[i]==1])).detach().numpy())
    waves_imag[i] = np.imag(wavelet(torch.Tensor(datas[i][Y[i]==1])).detach().numpy())

at_waves_real_amp = np.zeros(n_subjects)
at_waves_real_pic = np.zeros(n_subjects)
at_waves_imag_amp = np.zeros(n_subjects)
at_waves_imag_pic = np.zeros(n_subjects)
for i in range(n_subjects):
    at_waves_real_amp[i] = np.max(np.mean(waves_real[i][:,:,0,:],axis=0)) - np.min(np.mean(waves_real[i][:,:,0,:],axis=0))
    at_waves_real_pic[i] = np.argmax(np.abs(np.mean(np.mean(waves_real[i],axis=0)[:,0],axis=0)))
    at_waves_imag_amp[i] = np.max(np.mean(waves_imag[i][:,:,0,:],axis=0)) - np.min(np.mean(waves_imag[i][:,:,0,:],axis=0))
    at_waves_imag_pic[i] = np.argmax(np.abs(np.mean(np.mean(waves_imag[i],axis=0)[:,0],axis=0)))

at_waves_real_m = np.zeros(n_subjects)
at_all_cm_real = np.zeros((n_subjects,math.ceil(datas[0][Y[i]==1].shape[0]/8),math.ceil(datas[0][Y[i]==1].shape[0]/8)))
at_waves_imag_m = np.zeros(n_subjects)
all_cm_imag = np.zeros((n_subjects,math.ceil(datas[0][Y[i]==1].shape[0]/8),math.ceil(datas[0][Y[i]==1].shape[0]/8)))
for i in range(n_subjects):
    corr = stats.spearmanr(waves_real[i][:,:,channel,:].reshape(-1,22*waves_real.shape[-1]),axis=1)
    arr_reduced = block_reduce(corr.statistic, block_size=(8,8), func=np.median, cval=np.median(corr))
    at_waves_real_m[i] = np.median(arr_reduced)
    at_all_cm_real[i] = arr_reduced

    corr = stats.spearmanr(waves_imag[i][:,:,channel,:].reshape(-1,22*waves_imag.shape[-1]),axis=1)
    arr_reduced = block_reduce(corr.statistic, block_size=(8,8), func=np.median, cval=np.median(corr))
    at_waves_imag_m[i] = np.median(arr_reduced)
    all_cm_imag[i] = arr_reduced

########################### Save ###########################

bt_SNR_mean_mean = np.mean(bt_SNR,axis=1)
bt_SNR_std_mean = np.std(bt_SNR,axis=1)
bt_amp = bt_amp.copy()
bt_pic = bt_pic.copy()
bt_alphas_mean = np.mean(bt_alphas,axis=1)
bt_betas_mean = np.mean(bt_betas,axis=1)
bt_deltas_mean = np.mean(bt_deltas,axis=1)
bt_thetas_mean = np.mean(bt_thetas,axis=1)
# bt_corr_mean = bt_m.copy()
bt_corr_mean = np.mean([bt_all_cm[i][np.triu_indices_from(bt_all_cm[i], k=1)] for i in range(bt_all_cm.shape[0])],axis=1)

at_amp = at_amp.copy()
at_pic = at_pic.copy()
at_alphas_mean = np.mean(at_alphas,axis=1)
at_betas_mean = np.mean(at_betas,axis=1)
at_deltas_mean = np.mean(at_deltas,axis=1)
at_thetas_mean = np.mean(at_thetas,axis=1)
# at_corr_mean = at_m.copy()
at_corr_mean = np.mean([at_all_cm[i][np.triu_indices_from(at_all_cm[i], k=1)] for i in range(at_all_cm.shape[0])],axis=1)


bt_waves_real_amp = bt_waves_real_amp.copy()
bt_waves_real_pic = bt_waves_real_pic.copy()
bt_waves_real_corr_mean = bt_waves_real_m.copy()
bt_waves_imag_amp = bt_waves_imag_amp.copy()
bt_waves_imag_pic = bt_waves_imag_pic.copy()
bt_waves_imag_corr_mean = bt_waves_imag_m.copy()
bt_waves_amp = np.sqrt(bt_waves_real_amp**2+bt_waves_imag_amp**2)
bt_waves_pic = np.sqrt(bt_waves_real_pic**2+bt_waves_imag_pic**2)
bt_waves_corr_mean = np.sqrt(np.array([bt_all_cm_real[i][np.triu_indices_from(bt_all_cm_real[i], k=1)] for i in range(bt_all_cm_real.shape[0])])**2+
                                      np.array([bt_all_cm_imag_imag[i][np.triu_indices_from(bt_all_cm_imag_imag[i], k=1)] for i in range(bt_all_cm_imag_imag.shape[0])])**2)

at_waves_real_amp = at_waves_real_amp.copy()
at_waves_real_pic = at_waves_real_pic.copy()
at_waves_real_corr_mean = at_waves_real_m.copy()
at_waves_imag_amp = at_waves_imag_amp.copy()
at_waves_imag_pic = at_waves_imag_pic.copy()
at_waves_imag_corr_mean = at_waves_imag_m.copy()
at_waves_amp = np.sqrt(at_waves_real_amp**2+at_waves_imag_amp**2)
at_waves_pic = np.sqrt(at_waves_real_pic**2+at_waves_imag_pic**2)
at_waves_corr_mean = np.sqrt(np.array([at_all_cm_real[i][np.triu_indices_from(at_all_cm_real[i], k=1)] for i in range(at_all_cm_real.shape[0])])**2+
                                      np.array([all_cm_imag[i][np.triu_indices_from(all_cm_imag[i], k=1)] for i in range(all_cm_imag.shape[0])])**2)



measure_1point = {}
measure_1point["alphas"] = {}
measure_1point["alphas"]["BT"] = bt_alphas_mean
measure_1point["alphas"]["AT"] = at_alphas_mean
measure_1point["betas"] = {}
measure_1point["betas"]["BT"] = bt_betas_mean
measure_1point["betas"]["AT"] = at_betas_mean
measure_1point["deltas"] = {}
measure_1point["deltas"]["BT"] = bt_deltas_mean
measure_1point["deltas"]["AT"] = at_deltas_mean
measure_1point["thetas"] = {}
measure_1point["thetas"]["BT"] = bt_thetas_mean
measure_1point["thetas"]["AT"] = at_thetas_mean

measure_1point["amp"] = {}
measure_1point["amp"]["BT"] = bt_amp*1e6
measure_1point["amp"]["AT"] = at_amp
measure_1point["amp"]["Wave_BT"] = bt_waves_real_amp*1e6
measure_1point["amp"]["Wave_AT"] = at_waves_real_amp

measure_1point["pic"] = {}
measure_1point["pic"]["BT"] = bt_pic
measure_1point["pic"]["AT"] = at_pic
measure_1point["pic"]["Wave_BT"] = bt_waves_real_pic
measure_1point["pic"]["Wave_AT"] = at_waves_real_pic

measure_1point["corr_mean"] = {}
measure_1point["corr_mean"]["BT"] = bt_corr_mean
measure_1point["corr_mean"]["AT"] = at_corr_mean
measure_1point["corr_mean"]["Wave_BT"] = bt_waves_real_corr_mean
measure_1point["corr_mean"]["Wave_AT"] = at_waves_real_corr_mean

measure_1point["SNR_mean"] = {}
measure_1point["SNR_mean"]["BT"] = bt_SNR_mean_mean
measure_1point["SNR_std"] = {}
measure_1point["SNR_std"]["BT"] = bt_SNR_std_mean

pd.DataFrame(measure_1point).to_pickle("D:/s.velut/Documents/These/Protheus_PHD/Data/STL/ws0.35/time_sample/data/cal_2/1point_results.pkl")