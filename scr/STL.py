import mne
import argparse
from utils.EEG2CodeKeras import EEG2Code
import keras
from utils.Green_files.green.wavelet_layers import RealCovariance
import torch

from utils.Green_files.research_code.pl_utils import get_green
mne.set_log_level('ERROR')
from utils._utils import make_preds_accumul_aggresive
import numpy as np
import time
from pyriemann.estimation import  XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.metrics import balanced_accuracy_score,f1_score,recall_score, precision_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from utils.Alignments.aligner import Aligner
from utils.STLDataLoader import STLDataLoader
from utils.xdawn_supertrial import XdawnST
from imblearn.under_sampling import RandomUnderSampler

torch.set_default_device('cpu')

def balance(X,Y,domains):
    X_new = []
    Y_new = []
    domains_new = []
    if domains is not None:
        for d in np.unique(domains):
            ind_domain = np.where(domains==d)
            rus = RandomUnderSampler()
            counter=np.array(range(0,len(Y[ind_domain]))).reshape(-1,1)
            index,_ = rus.fit_resample(counter,Y[ind_domain])
            index = np.sort(index,axis=0)
            X_new.append(np.squeeze(X[ind_domain][index,:,:], axis=1))
            Y_new.append(np.squeeze(Y[ind_domain][index]))
            domains_new.append(np.squeeze(domains[ind_domain][index]))
        return np.concatenate(X_new),np.concatenate(Y_new),np.concatenate(domains_new)
    else:
        rus = RandomUnderSampler()
        counter=np.array(range(0,len(Y))).reshape(-1,1)
        index,_ = rus.fit_resample(counter,Y)
        index = np.sort(index,axis=0)
        X = np.squeeze(X[index,:,:], axis=1)
        Y = np.squeeze(Y[index])
        return X,Y,None

def get_all_metrics(Y_test, Y_pred, code_test, code_pred):

    score = balanced_accuracy_score(Y_test,Y_pred)
    recall = recall_score(Y_test,Y_pred)
    f1 = f1_score(Y_test,Y_pred)
    precision = precision_score(Y_test,Y_pred)

    score_code = balanced_accuracy_score(code_test,code_pred)

    return score, recall, f1, score_code,precision

def full_preprocessed_data(path,participant):
    return np.load(path+"full_preprocess_data_"+participant+".npy")


def solo_preprocessed_data(path,participant):
    return np.load(path+"full_solo_preprocess_data_"+participant+".npy")

def perform_measure_TF(clf, X_train, Y_train, X_test, Y_test, codes, n_class=5, n_cal=4, window_size=0.35, freqwise=500,
                    test_size=0.2, batchsize= 64, lr=1e-3, num_epochs=20, device=torch.device("cpu")):
    # Get the training data 

    print("Fitting")
    start = time.time()
    clf = clf.fit(X_train, Y_train)


    tps_train = time.time() - start

    temp_start = time.time()
    y_pred = clf.predict(X_test)
    y_pred = np.array(y_pred)
    Y_pred = np.array([1 if (y >= 0.5) else 0 for y in y_pred])

    tps_pred = time.time() - temp_start

    labels_pred_accumul, code_buffer, mean_long_accumul = make_preds_accumul_aggresive(
            Y_pred, codes, min_len=30, sfreq=freqwise, consecutive=50, window_size=window_size
        )
    
    tps_acc = np.mean(mean_long_accumul)

    return Y_test, Y_pred, labels_pred_accumul, tps_train, tps_pred, tps_acc, clf

def get_train_test_data(Xt, Yt, Xs, Ys, domainst, domainss, codes, labels_code, method, clf_name, n_class=5, n_cal=2, window_size=0.35, freqwise=500,
                    test_size=0.2):
    if method=="SiSu":
        # Initialisation
        nb_sample_cal = int(n_class*n_cal*(2.2-window_size)*freqwise)

        # Get the training data 
        X_train = Xt[:nb_sample_cal]
        Y_train = Yt[:nb_sample_cal]
        domains_train = domainst[:nb_sample_cal]
        X_test = Xt[nb_sample_cal:]
        Y_test = Yt[nb_sample_cal:]
        domains_test = domainst[nb_sample_cal:]
        labels_code_test = labels_code[(n_class*n_cal):]
        if clf_name in ["TS_LDA","CNN","GREEN"]:
            X_std = Xt[:nb_sample_cal].std(axis=0)
            X_train = X_train/(X_std + 1e-8)
            X_test = X_test/(X_std + 1e-8)
        X_train, Y_train, domains_train = balance(X_train,Y_train,domains_train)
    elif method=="DA":
        # Initialisation
        nb_sample_cal = int(n_class*n_cal*(2.2-window_size)*freqwise)

        # Get the training data 
        X_train = np.concatenate([Xs,Xt[:nb_sample_cal]]).reshape(-1,Xs.shape[-2],Xs.shape[-1])
        Y_train = np.concatenate([Ys,Yt[:nb_sample_cal]]).reshape(-1)
        domains_train = np.concatenate([domainss,domainst[:nb_sample_cal]]).reshape(-1)
        X_test = Xt[nb_sample_cal:]
        Y_test = Yt[nb_sample_cal:]
        domains_test = domainst[nb_sample_cal:]
        labels_code_test = labels_code[(n_class*n_cal):]
        if clf_name in ["TS_LDA","CNN","GREEN"]:
            X_std = Xt[:nb_sample_cal].std(axis=0)
            X_train = X_train/(X_std + 1e-8)
            X_test = X_test/(X_std + 1e-8)

        X_train, Y_train, domains_train = balance(X_train,Y_train,domains_train)

    return X_train, Y_train, X_test, Y_test, domains_train, domains_test, labels_code_test


def main(path, file_path, fmin, fmax, sample_freq, fps, timewise, participants, clf_name,
         method="DA", window_size=0.35, test_size=0.2, batchsize=64, lr=1e-3, num_epochs=20, prefix=''):
    ##### Here is to centralised main steps
    participants = eval(participants)

    data_path = '/'.join([file_path,"ws"+str(window_size),timewise,"data",''])

    # get the data
    dl = STLDataLoader(path, fmin, fmax, window_size, sample_freq, fps, timewise, participants, 5)
    raw_data = dl.load_data()
    X, Y, domains, codes, labels_code = dl.get_epochs(raw_data)

    # Initialisation
    all_metrics = np.zeros((9,dl.nb_subjects, dl.nb_subjects))
    # Start for loop on each participant (or just on a few to go faster)
    for i in range(dl.nb_subjects):
        print("Check participant ",i)
        
        # Preprocess the data for data i with no data leak(index of the loop). HERE IF DOMAIN ADAPTATION
        tps_preproc = 0
        # Start another for loop to perform 2by2 measure (similarity, Domain Adaptation/Generalisation)
        for j in range(dl.nb_subjects):
            print("With participant", j)
            if (method!="SiSu" and j!=i) or (method=="SiSu" and j==i):

                
                if clf_name in ['PTGREEN','PTCNN']:
                    if method in ["DA","SiSu"]:
                        print("Preprocess the data of participant ",i)
                        temp_start = time.time()
                        Xt_preproc = solo_preprocessed_data(data_path,participants[i])
                        tps_preproc = time.time() - temp_start
                    else:
                        Xt_preproc=None
                    X_preproc = full_preprocessed_data(data_path,participants[j])
                    X_train, Y_train, X_test, Y_test, domains_train, domains_test, labels_code_test = get_train_test_data(Xt_preproc, Y[i], X_preproc,
                                                                                Y[j], domains[i], domains[j], codes, 
                                                                                labels_code[i], method, clf_name, dl.n_class,2, window_size,
                                                                                dl.freqwise, test_size)
                    if clf_name=='PTGREEN':
                        model = get_green(
                                    n_freqs=22,
                                    kernel_width_s=window_size,
                                    n_ch=8,
                                    sfreq=500,
                                    oct_min=0,
                                    oct_max=4.4,
                                    orth_weights=False,
                                    dropout=.6,
                                    hidden_dim=[20,10],
                                    logref='logeuclid',
                                    pool_layer=RealCovariance(),
                                    bi_out=[4],
                                    dtype=torch.float32,
                                    out_dim=2,
                                    )
                    elif clf_name=='PTCNN':
                        optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
                        model = EEG2Code(windows_size = X_train[0].shape[-1],
                                         n_channel_input = X_train[0].shape[-2],
                                         optimizer=optimizer,
                                         num_epochs=num_epochs)
                    
                    # perform the train test with Xsource and Xtarget
                    print("Train and test")
                    Y_test, Y_pred, labels_pred_accumul, tps_train, tps_pred, tps_acc, clf = perform_measure_TF(model, X_train, Y_train, X_test, Y_test,
                                                                                                                    codes, dl.n_class,2, window_size,
                                                                                                                    dl.freqwise,test_size,batchsize,lr,
                                                                                                                    num_epochs)

                    
                # Create the classifier
                # if clf_name in ["TS_LDA","TS_SVM","MDM","CCNN","CNN","CGREEN","GREEN","DACNN"]:
                else:
                    X_train, Y_train, X_test, Y_test, domains_train, domains_test, labels_code_test = get_train_test_data(X[i], Y[i], X[j],
                                                                                Y[j], domains[i], domains[j], codes, 
                                                                                labels_code[i], method, clf_name, dl.n_class,2, window_size,
                                                                                dl.freqwise, test_size)

                    X_std = X_train.std(axis=0)
                    X_train = X_train/(X_std + 1e-8)
                    X_test = X_test/(X_std + 1e-8)
                    if clf_name=='CGREEN':
                        model = make_pipeline(
                                XdawnST(nfilter=4,classes=[1],estimator='lwf'),
                                Aligner(estimator="lwf",metric="real"),
                                get_green(
                                    n_freqs=22,
                                    kernel_width_s=window_size,
                                    n_ch=8,
                                    sfreq=500,
                                    oct_min=0,
                                    oct_max=4.4,
                                    orth_weights=False,
                                    dropout=.6,
                                    hidden_dim=[20,10],
                                    logref='logeuclid',
                                    pool_layer=RealCovariance(),
                                    bi_out=[4],
                                    dtype=torch.float32,
                                    out_dim=2,
                                    )
                                )
                    if clf_name=='GREEN':
                        model = get_green(
                                    n_freqs=22,
                                    kernel_width_s=window_size,
                                    n_ch=8,
                                    sfreq=500,
                                    oct_min=0,
                                    oct_max=4.4,
                                    orth_weights=False,
                                    dropout=.6,
                                    hidden_dim=[20,10],
                                    logref='logeuclid',
                                    pool_layer=RealCovariance(),
                                    bi_out=[4],
                                    dtype=torch.float32,
                                    out_dim=2,
                                    )
                    if clf_name=="TS_LDA":
                        model = make_pipeline(XdawnCovariances(nfilter=4,xdawn_estimator="lwf",estimator="lwf",classes=[1]),
                            TangentSpace(), LDA(solver="lsqr", shrinkage="auto"))
                    elif clf_name=="CNN":
                        optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
                        model = EEG2Code(windows_size = X_train[0].shape[-1],
                                         n_channel_input = X_train[0].shape[-2],
                                         optimizer=optimizer,
                                         num_epochs=num_epochs)
                    elif clf_name=="CCNN":
                        optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
                        model = make_pipeline(
                                XdawnST(nfilter=4,classes=[1],estimator='lwf'),
                                Aligner(estimator="lwf",metric="real"),EEG2Code(windows_size = X_train[0].shape[-1],
                                         n_channel_input = X_train[0].shape[-2],
                                         optimizer=optimizer,
                                         num_epochs=num_epochs)
                        )
                    

                    # perform the train test with Xsource and Xtarget
                    print("Train and test")
                    Y_test, Y_pred, labels_pred_accumul, tps_train, tps_pred, tps_acc, clf = perform_measure_TF(model, X_train, Y_train, X_test, Y_test,
                                                                                                                    codes, dl.n_class,2, window_size,
                                                                                                                    dl.freqwise,test_size,batchsize,lr,
                                                                                                                    num_epochs)

                # Calcul the different classification metric 
                score, recall, f1, score_code,precision = get_all_metrics(Y_test, Y_pred, labels_code_test, labels_pred_accumul)
                print("score_code",score_code)
                all_metrics[:,i,j] = np.array([tps_preproc, tps_train, tps_pred, tps_acc, score, recall, f1, score_code,precision])

    save_path = '/'.join([file_path,"ws"+str(window_size),timewise,"results","cal_2",''])
    name = ["tps_preproc_"+prefix+".npy", "tps_train_"+prefix+".npy", "tps_pred_"+prefix+".npy", "tps_acc_"+prefix+".npy", "score_"+prefix+".npy","recall_"+prefix+".npy", "f1_"+prefix+".npy", "score_code_"+prefix+".npy","precision_"+prefix+".npy"]
    for i in range(all_metrics.shape[0]):
        np.save(save_path+name[i],all_metrics[i])
        print(i,all_metrics[i])

    return all_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants",default="['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24']", help="The index of the subject to get data from")
    parser.add_argument("--timewise",default="time_sample", help="The index of the subject to test on")
    parser.add_argument("--clf_name",default="1",help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--ws",default=0.35,type=float,help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--nb_epoch",default=20,type=int,help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--path",default='../Data/Dry_Ricker/',help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--fpath",default='../Data/results',help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--method",default='DA',help="Boolean to recenter the data before classifying or not")

    # path = 'D:/s.velut/Documents/These/Protheus_PHD/Data/Dry_Ricker/'
    # file_path = 'D:/s.velut/Documents/These/Protheus_PHD/Data/STL'
    # n_class=5
    # fmin = 1
    # fmax = 45
    # fps = 60
    # window_size = 0.35
    # sfreq = 500
    # num_epochs = 20
    # timewise="time_sample"
    # clf_name = "CNN"
    # method = "SiSu"
    # # participants = '["P1","P2"]'
    # participants = '["P1","P17","P16","P19","P15","P23"]'
    # # participants = "['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10',\
    # #                 'P11','P12','P13','P14','P15','P16','P17','P18','P19','P20',\
    # #                 'P21','P22','P23','P24']"
    # test_size=0.2
    # batchsize=64
    # lr=1e-03

    # args = parser.parse_args()
    # prefix = clf_name+method+""

    # sim, metric = main(path,file_path,fmin,fmax,sfreq,fps,timewise,participants,clf_name,
    #                    method,window_size,test_size,batchsize,lr,num_epochs=num_epochs,prefix=prefix)

    n_class=5
    fmin = 1
    fmax = 45
    fps = 60
    sfreq = 500
    test_size=0.2
    batchsize=64
    lr=1e-03

    args = parser.parse_args()
    prefix = args.clf_name+args.method+""

    sim, metric = main(args.path,args.fpath,fmin,fmax,sfreq,fps,args.timewise,args.participants,args.clf_name,
                       args.method,args.ws,test_size,batchsize,lr,num_epochs=args.nb_epoch,prefix=prefix)
                       