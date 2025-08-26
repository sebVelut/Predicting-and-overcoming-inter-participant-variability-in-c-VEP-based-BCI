import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import PtitPrince as pt
from sklearn import linear_model
from scipy import stats
from matplotlib import rcParams



files_path = "../results/ws0.35/time_sample/results/cal_2/"
participants = np.array(['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10',
                'P11','P12','P13','P14','P15','P16','P17','P18','P19','P20',
                'P21','P22','P23','P24'])
name = ["tps_preproc", "tps_train", "tps_pred", "tps_acc", "score", "recall", "f1", "score_code"]

######  GREEN   ###########

tps_preproc_GREENSiSu = np.load(files_path+"tps_preproc_GREENSiSu.npy")
tps_train_GREENSiSu = np.load(files_path+"tps_train_GREENSiSu.npy")
tps_pred_GREENSiSu = np.load(files_path+"tps_pred_GREENSiSu.npy")
tps_acc_GREENSiSu = np.load(files_path+"tps_acc_GREENSiSu.npy")
score_GREENSiSu = np.load(files_path+"score_GREENSiSu.npy")
recall_GREENSiSu = np.load(files_path+"recall_GREENSiSu.npy")
f1_GREENSiSu = np.load(files_path+"f1_GREENSiSu.npy")
score_code_GREENSiSu = np.load(files_path+"score_code_GREENSiSu.npy")

tps_preproc_GREEN = np.load(files_path+"tps_preproc_GREENDA.npy")
tps_train_GREEN = np.load(files_path+"tps_train_GREENDA.npy")
tps_pred_GREEN = np.load(files_path+"tps_pred_GREENDA.npy")
tps_acc_GREEN = np.load(files_path+"tps_acc_GREENDA.npy")
score_GREEN = np.load(files_path+"score_GREENDA.npy")
recall_GREEN = np.load(files_path+"recall_GREENDA.npy")
f1_GREEN = np.load(files_path+"f1_GREENDA.npy")
score_code_GREEN = np.load(files_path+"score_code_GREENDA.npy")

np.fill_diagonal(tps_preproc_GREEN,np.diagonal(tps_preproc_GREENSiSu))
np.fill_diagonal(tps_train_GREEN,np.diagonal(tps_train_GREENSiSu))
np.fill_diagonal(tps_pred_GREEN,np.diagonal(tps_pred_GREENSiSu))
np.fill_diagonal(tps_acc_GREEN,np.diagonal(tps_acc_GREENSiSu))
np.fill_diagonal(score_GREEN,np.diagonal(score_GREENSiSu))
np.fill_diagonal(recall_GREEN,np.diagonal(recall_GREENSiSu))
np.fill_diagonal(f1_GREEN,np.diagonal(f1_GREENSiSu))
np.fill_diagonal(score_code_GREEN,np.diagonal(score_code_GREENSiSu))

###### PTGREEN ###########

tps_preproc_PTGREENSiSu = np.load(files_path+"tps_preproc_PTGREENSiSu.npy")
tps_train_PTGREENSiSu = np.load(files_path+"tps_train_PTGREENSiSu.npy")
tps_pred_PTGREENSiSu = np.load(files_path+"tps_pred_PTGREENSiSu.npy")
tps_acc_PTGREENSiSu = np.load(files_path+"tps_acc_PTGREENSiSu.npy")
score_PTGREENSiSu = np.load(files_path+"score_PTGREENSiSu.npy")
recall_PTGREENSiSu = np.load(files_path+"recall_PTGREENSiSu.npy")
f1_PTGREENSiSu = np.load(files_path+"f1_PTGREENSiSu.npy")
score_code_PTGREENSiSu = np.load(files_path+"score_code_PTGREENSiSu.npy")

tps_preproc_PTGREEN = np.load(files_path+"tps_preproc_PTGREENDA.npy")
tps_train_PTGREEN = np.load(files_path+"tps_train_PTGREENDA.npy")
tps_pred_PTGREEN = np.load(files_path+"tps_pred_PTGREENDA.npy")
tps_acc_PTGREEN = np.load(files_path+"tps_acc_PTGREENDA.npy")
score_PTGREEN = np.load(files_path+"score_PTGREENDA.npy")
recall_PTGREEN = np.load(files_path+"recall_PTGREENDA.npy")
f1_PTGREEN = np.load(files_path+"f1_PTGREENDA.npy")
score_code_PTGREEN = np.load(files_path+"score_code_PTGREENDA.npy")

np.fill_diagonal(tps_preproc_PTGREEN,np.diagonal(tps_preproc_PTGREENSiSu))
np.fill_diagonal(tps_train_PTGREEN,np.diagonal(tps_train_PTGREENSiSu))
np.fill_diagonal(tps_pred_PTGREEN,np.diagonal(tps_pred_PTGREENSiSu))
np.fill_diagonal(tps_acc_PTGREEN,np.diagonal(tps_acc_PTGREENSiSu))
np.fill_diagonal(score_PTGREEN,np.diagonal(score_PTGREENSiSu))
np.fill_diagonal(recall_PTGREEN,np.diagonal(recall_PTGREENSiSu))
np.fill_diagonal(f1_PTGREEN,np.diagonal(f1_PTGREENSiSu))
np.fill_diagonal(score_code_PTGREEN,np.diagonal(score_code_PTGREENSiSu))

######  CGREEN   ###########

tps_preproc_CGREENSiSu = np.load(files_path+"tps_preproc_CGREENSiSu.npy")
tps_train_CGREENSiSu = np.load(files_path+"tps_train_CGREENSiSu.npy")
tps_pred_CGREENSiSu = np.load(files_path+"tps_pred_CGREENSiSu.npy")
tps_acc_CGREENSiSu = np.load(files_path+"tps_acc_CGREENSiSu.npy")
score_CGREENSiSu = np.load(files_path+"score_CGREENSiSu.npy")
recall_CGREENSiSu = np.load(files_path+"recall_CGREENSiSu.npy")
f1_CGREENSiSu = np.load(files_path+"f1_CGREENSiSu.npy")
score_code_CGREENSiSu = np.load(files_path+"score_code_CGREENSiSu.npy")

tps_preproc_CGREEN = np.load(files_path+"tps_preproc_CGREENDA.npy")
tps_train_CGREEN = np.load(files_path+"tps_train_CGREENDA.npy")
tps_pred_CGREEN = np.load(files_path+"tps_pred_CGREENDA.npy")
tps_acc_CGREEN = np.load(files_path+"tps_acc_CGREENDA.npy")
score_CGREEN = np.load(files_path+"score_CGREENDA.npy")
recall_CGREEN = np.load(files_path+"recall_CGREENDA.npy")
f1_CGREEN = np.load(files_path+"f1_CGREENDA.npy")
score_code_CGREEN = np.load(files_path+"score_code_CGREENDA.npy")

np.fill_diagonal(tps_train_CGREEN,np.diagonal(tps_train_CGREENSiSu))
np.fill_diagonal(tps_pred_CGREEN,np.diagonal(tps_pred_CGREENSiSu))
np.fill_diagonal(tps_acc_CGREEN,np.diagonal(tps_acc_CGREENSiSu))
np.fill_diagonal(score_CGREEN,np.diagonal(score_CGREENSiSu))
np.fill_diagonal(recall_CGREEN,np.diagonal(recall_CGREENSiSu))
np.fill_diagonal(f1_CGREEN,np.diagonal(f1_CGREENSiSu))
np.fill_diagonal(score_code_CGREEN,np.diagonal(score_code_CGREENSiSu))

######  TSLDA   ###########

tps_preproc_TSLDA = np.load(files_path+"tps_preproc_TSLDADA.npy")
tps_train_TSLDA = np.load(files_path+"tps_train_TSLDADA.npy")
tps_pred_TSLDA = np.load(files_path+"tps_pred_TSLDADA.npy")
tps_acc_TSLDA = np.load(files_path+"tps_acc_TSLDADA.npy")
score_TSLDA = np.load(files_path+"score_TSLDADA.npy")
recall_TSLDA = np.load(files_path+"recall_TSLDADA.npy")
f1_TSLDA = np.load(files_path+"f1_TSLDADA.npy")
score_code_TSLDA = np.load(files_path+"score_code_TSLDADA.npy")

tps_preproc_TSLDASiSu = np.load(files_path+"tps_preproc_TS_LDASiSu.npy")
tps_train_TSLDASiSu = np.load(files_path+"tps_train_TS_LDASiSu.npy")
tps_pred_TSLDASiSu = np.load(files_path+"tps_pred_TS_LDASiSu.npy")
tps_acc_TSLDASiSu = np.load(files_path+"tps_acc_TS_LDASiSu.npy")
score_TSLDASiSu = np.load(files_path+"score_TS_LDASiSu.npy")
recall_TSLDASiSu = np.load(files_path+"recall_TS_LDASiSu.npy")
f1_TSLDASiSu = np.load(files_path+"f1_TS_LDASiSu.npy")
score_code_TSLDASiSu = np.load(files_path+"score_code_TS_LDASiSu.npy")

np.fill_diagonal(tps_preproc_TSLDA,np.diagonal(tps_preproc_TSLDASiSu))
np.fill_diagonal(tps_train_TSLDA,np.diagonal(tps_train_TSLDASiSu))
np.fill_diagonal(tps_pred_TSLDA,np.diagonal(tps_pred_TSLDASiSu))
np.fill_diagonal(tps_acc_TSLDA,np.diagonal(tps_acc_TSLDASiSu))
np.fill_diagonal(score_TSLDA,np.diagonal(score_TSLDASiSu))
np.fill_diagonal(recall_TSLDA,np.diagonal(recall_TSLDASiSu))
np.fill_diagonal(f1_TSLDA,np.diagonal(f1_TSLDASiSu))
np.fill_diagonal(score_code_TSLDA,np.diagonal(score_code_TSLDASiSu))


#######  CNN   ###########

tps_preproc_CNN = np.load(files_path+"tps_preproc_CNNDA.npy")
tps_train_CNN = np.load(files_path+"tps_train_CNNDA.npy")
tps_pred_CNN = np.load(files_path+"tps_pred_CNNDA.npy")
tps_acc_CNN = np.load(files_path+"tps_acc_CNNDA.npy")
score_CNN = np.load(files_path+"score_CNNDA.npy")
recall_CNN = np.load(files_path+"recall_CNNDA.npy")
f1_CNN = np.load(files_path+"f1_CNNDA.npy")
score_code_CNN = np.load(files_path+"score_code_CNNDA.npy")

tps_preproc_CNNSiSu = np.load(files_path+"tps_preproc_CNNSiSu.npy")
tps_train_CNNSiSu = np.load(files_path+"tps_train_CNNSiSu.npy")
tps_pred_CNNSiSu = np.load(files_path+"tps_pred_CNNSiSu.npy")
tps_acc_CNNSiSu = np.load(files_path+"tps_acc_CNNSiSu.npy")
score_CNNSiSu = np.load(files_path+"score_CNNSiSu.npy")
recall_CNNSiSu = np.load(files_path+"recall_CNNSiSu.npy")
f1_CNNSiSu = np.load(files_path+"f1_CNNSiSu.npy")
score_code_CNNSiSu = np.load(files_path+"score_code_CNNSiSu.npy")

np.fill_diagonal(tps_preproc_CNN,np.diagonal(tps_preproc_CNNSiSu))
np.fill_diagonal(tps_train_CNN,np.diagonal(tps_train_CNNSiSu))
np.fill_diagonal(tps_pred_CNN,np.diagonal(tps_pred_CNNSiSu))
np.fill_diagonal(tps_acc_CNN,np.diagonal(tps_acc_CNNSiSu))
np.fill_diagonal(score_CNN,np.diagonal(score_CNNSiSu))
np.fill_diagonal(recall_CNN,np.diagonal(recall_CNNSiSu))
np.fill_diagonal(f1_CNN,np.diagonal(f1_CNNSiSu))
np.fill_diagonal(score_code_CNN,np.diagonal(score_code_CNNSiSu))

#######  PTCNN   ###########

tps_preproc_PTCNN = np.load(files_path+"tps_preproc_PTCNNDA.npy")
tps_train_PTCNN = np.load(files_path+"tps_train_PTCNNDA.npy")
tps_pred_PTCNN = np.load(files_path+"tps_pred_PTCNNDA.npy")
tps_acc_PTCNN = np.load(files_path+"tps_acc_PTCNNDA.npy")
score_PTCNN = np.load(files_path+"score_PTCNNDA.npy")
recall_PTCNN = np.load(files_path+"recall_PTCNNDA.npy")
f1_PTCNN = np.load(files_path+"f1_PTCNNDA.npy")
score_code_PTCNN = np.load(files_path+"score_code_PTCNNDA.npy")

tps_preproc_PTCNNSiSu = np.load(files_path+"tps_preproc_PTCNNSiSu.npy")
tps_train_PTCNNSiSu = np.load(files_path+"tps_train_PTCNNSiSu.npy")
tps_pred_PTCNNSiSu = np.load(files_path+"tps_pred_PTCNNSiSu.npy")
tps_acc_PTCNNSiSu = np.load(files_path+"tps_acc_PTCNNSiSu.npy")
score_PTCNNSiSu = np.load(files_path+"score_PTCNNSiSu.npy")
recall_PTCNNSiSu = np.load(files_path+"recall_PTCNNSiSu.npy")
f1_PTCNNSiSu = np.load(files_path+"f1_PTCNNSiSu.npy")
score_code_PTCNNSiSu = np.load(files_path+"score_code_PTCNNSiSu.npy")

np.fill_diagonal(tps_preproc_PTCNN,np.diagonal(tps_preproc_PTCNNSiSu))
np.fill_diagonal(tps_train_PTCNN,np.diagonal(tps_train_PTCNNSiSu))
np.fill_diagonal(tps_pred_PTCNN,np.diagonal(tps_pred_PTCNNSiSu))
np.fill_diagonal(tps_acc_PTCNN,np.diagonal(tps_acc_PTCNNSiSu))
np.fill_diagonal(score_PTCNN,np.diagonal(score_PTCNNSiSu))
np.fill_diagonal(recall_PTCNN,np.diagonal(recall_PTCNNSiSu))
np.fill_diagonal(f1_PTCNN,np.diagonal(f1_PTCNNSiSu))
np.fill_diagonal(score_code_PTCNN,np.diagonal(score_code_PTCNNSiSu))

#######  CCNN   ###########

tps_preproc_CCNN = np.load(files_path+"tps_preproc_CCNNDA.npy")
tps_train_CCNN = np.load(files_path+"tps_train_CCNNDA.npy")
tps_pred_CCNN = np.load(files_path+"tps_pred_CCNNDA.npy")
tps_acc_CCNN = np.load(files_path+"tps_acc_CCNNDA.npy")
score_CCNN = np.load(files_path+"score_CCNNDA.npy")
recall_CCNN = np.load(files_path+"recall_CCNNDA.npy")
f1_CCNN = np.load(files_path+"f1_CCNNDA.npy")
score_code_CCNN = np.load(files_path+"score_code_CCNNDA.npy")

tps_preproc_CCNNSiSu = np.load(files_path+"tps_preproc_CCNNSiSu.npy")
tps_train_CCNNSiSu = np.load(files_path+"tps_train_CCNNSiSu.npy")
tps_pred_CCNNSiSu = np.load(files_path+"tps_pred_CCNNSiSu.npy")
tps_acc_CCNNSiSu = np.load(files_path+"tps_acc_CCNNSiSu.npy")
score_CCNNSiSu = np.load(files_path+"score_CCNNSiSu.npy")
recall_CCNNSiSu = np.load(files_path+"recall_CCNNSiSu.npy")
f1_CCNNSiSu = np.load(files_path+"f1_CCNNSiSu.npy")
score_code_CCNNSiSu = np.load(files_path+"score_code_CCNNSiSu.npy")

np.fill_diagonal(tps_preproc_CCNN,np.diagonal(tps_preproc_CCNNSiSu))
np.fill_diagonal(tps_train_CCNN,np.diagonal(tps_train_CCNNSiSu))
np.fill_diagonal(tps_pred_CCNN,np.diagonal(tps_pred_CCNNSiSu))
np.fill_diagonal(tps_acc_CCNN,np.diagonal(tps_acc_CCNNSiSu))
np.fill_diagonal(score_CCNN,np.diagonal(score_CCNNSiSu))
np.fill_diagonal(recall_CCNN,np.diagonal(recall_CCNNSiSu))
np.fill_diagonal(f1_CCNN,np.diagonal(f1_CCNNSiSu))
np.fill_diagonal(score_code_CCNN,np.diagonal(score_code_CCNNSiSu))

##############################################


df_score = pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-GREEN","score":np.ndarray.flatten(score_CGREEN[~np.eye(score_CGREEN.shape[0],dtype=bool)].reshape(score_CGREEN.shape[0],-1)),"mode":"DA"})
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"GREEN","score":np.ndarray.flatten(np.ndarray.flatten(score_GREEN[~np.eye(score_GREEN.shape[0],dtype=bool)].reshape(score_GREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-GREEN","score":np.ndarray.flatten(np.ndarray.flatten(score_PTGREEN[~np.eye(score_PTGREEN.shape[0],dtype=bool)].reshape(score_PTGREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"TS-LDA","score":np.ndarray.flatten(np.ndarray.flatten(score_TSLDA[~np.eye(score_TSLDA.shape[0],dtype=bool)].reshape(score_TSLDA.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-CNN","score":np.ndarray.flatten(np.ndarray.flatten(score_CCNN[~np.eye(score_CCNN.shape[0],dtype=bool)].reshape(score_CCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"CNN","score":np.ndarray.flatten(np.ndarray.flatten(score_CNN[~np.eye(score_CNN.shape[0],dtype=bool)].reshape(score_CNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-CNN","score":np.ndarray.flatten(np.ndarray.flatten(score_PTCNN[~np.eye(score_PTCNN.shape[0],dtype=bool)].reshape(score_PTCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-GREEN","score":np.diagonal(score_CGREENSiSu),"mode":"WP"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"GREEN","score":np.diagonal(score_GREENSiSu),"mode":"WP"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-GREEN","score":np.diagonal(score_PTGREENSiSu),"mode":"WP"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"TS-LDA","score":np.diagonal(score_TSLDASiSu),"mode":"WP"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-CNN","score":np.diagonal(score_CCNNSiSu),"mode":"WP"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"CNN","score":np.diagonal(score_CNNSiSu),"mode":"WP"})],ignore_index=True)
df_score = pd.concat([df_score,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-CNN","score":np.diagonal(score_PTCNNSiSu),"mode":"WP"})],ignore_index=True)

df_score_code = pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-GREEN","score":np.ndarray.flatten(score_code_CGREEN[~np.eye(score_code_CGREEN.shape[0],dtype=bool)].reshape(score_code_CGREEN.shape[0],-1)),"mode":"DA"})
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"GREEN","score":np.ndarray.flatten(np.ndarray.flatten(score_code_GREEN[~np.eye(score_code_GREEN.shape[0],dtype=bool)].reshape(score_code_GREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-GREEN","score":np.ndarray.flatten(np.ndarray.flatten(score_code_PTGREEN[~np.eye(score_code_PTGREEN.shape[0],dtype=bool)].reshape(score_code_PTGREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"TS-LDA","score":np.ndarray.flatten(np.ndarray.flatten(score_code_TSLDA[~np.eye(score_code_TSLDA.shape[0],dtype=bool)].reshape(score_code_TSLDA.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-CNN","score":np.ndarray.flatten(np.ndarray.flatten(score_code_CCNN[~np.eye(score_code_CCNN.shape[0],dtype=bool)].reshape(score_code_CCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"CNN","score":np.ndarray.flatten(np.ndarray.flatten(score_code_CNN[~np.eye(score_code_CNN.shape[0],dtype=bool)].reshape(score_code_CNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-CNN","score":np.ndarray.flatten(np.ndarray.flatten(score_code_PTCNN[~np.eye(score_code_PTCNN.shape[0],dtype=bool)].reshape(score_code_PTCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-GREEN","score":np.diagonal(score_code_CGREENSiSu),"mode":"WP"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"GREEN","score":np.diagonal(score_code_GREENSiSu),"mode":"WP"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-GREEN","score":np.diagonal(score_code_PTGREENSiSu),"mode":"WP"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"TS-LDA","score":np.diagonal(score_code_TSLDASiSu),"mode":"WP"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-CNN","score":np.diagonal(score_code_CCNNSiSu),"mode":"WP"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"CNN","score":np.diagonal(score_code_CNNSiSu),"mode":"WP"})],ignore_index=True)
df_score_code = pd.concat([df_score_code,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-CNN","score":np.diagonal(score_code_PTCNNSiSu),"mode":"WP"})],ignore_index=True)

df_tps_train = pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-GREEN","score":np.ndarray.flatten(tps_train_CGREEN[~np.eye(tps_train_CGREEN.shape[0],dtype=bool)].reshape(tps_train_CGREEN.shape[0],-1)),"mode":"DA"})
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"GREEN","score":np.ndarray.flatten(np.ndarray.flatten(tps_train_GREEN[~np.eye(tps_train_GREEN.shape[0],dtype=bool)].reshape(tps_train_GREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-GREEN","score":np.ndarray.flatten(np.ndarray.flatten(tps_train_PTGREEN[~np.eye(tps_train_PTGREEN.shape[0],dtype=bool)].reshape(tps_train_PTGREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"TS-LDA","score":np.ndarray.flatten(np.ndarray.flatten(tps_train_TSLDA[~np.eye(tps_train_TSLDA.shape[0],dtype=bool)].reshape(tps_train_TSLDA.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-CNN","score":np.ndarray.flatten(np.ndarray.flatten(tps_train_CCNN[~np.eye(tps_train_CCNN.shape[0],dtype=bool)].reshape(tps_train_CCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"CNN","score":np.ndarray.flatten(np.ndarray.flatten(tps_train_CNN[~np.eye(tps_train_CNN.shape[0],dtype=bool)].reshape(tps_train_CNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-CNN","score":np.ndarray.flatten(np.ndarray.flatten(tps_train_PTCNN[~np.eye(tps_train_PTCNN.shape[0],dtype=bool)].reshape(tps_train_PTCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-GREEN","score":np.diagonal(tps_train_CGREENSiSu),"mode":"WP"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"GREEN","score":np.diagonal(tps_train_GREENSiSu),"mode":"WP"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-GREEN","score":np.diagonal(tps_train_PTGREENSiSu),"mode":"WP"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"TS-LDA","score":np.diagonal(tps_train_TSLDASiSu),"mode":"WP"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-CNN","score":np.diagonal(tps_train_CCNNSiSu),"mode":"WP"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"CNN","score":np.diagonal(tps_train_CNNSiSu),"mode":"WP"})],ignore_index=True)
df_tps_train = pd.concat([df_tps_train,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-CNN","score":np.diagonal(tps_train_PTCNNSiSu),"mode":"WP"})],ignore_index=True)

df_tps_pred = pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-GREEN","score":np.ndarray.flatten(tps_pred_CGREEN[~np.eye(tps_pred_CGREEN.shape[0],dtype=bool)].reshape(tps_pred_CGREEN.shape[0],-1)),"mode":"DA"})
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"GREEN","score":np.ndarray.flatten(np.ndarray.flatten(tps_pred_GREEN[~np.eye(tps_pred_GREEN.shape[0],dtype=bool)].reshape(tps_pred_GREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-GREEN","score":np.ndarray.flatten(np.ndarray.flatten(tps_pred_PTGREEN[~np.eye(tps_pred_PTGREEN.shape[0],dtype=bool)].reshape(tps_pred_PTGREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"TS-LDA","score":np.ndarray.flatten(np.ndarray.flatten(tps_pred_TSLDA[~np.eye(tps_pred_TSLDA.shape[0],dtype=bool)].reshape(tps_pred_TSLDA.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-CNN","score":np.ndarray.flatten(np.ndarray.flatten(tps_pred_CCNN[~np.eye(tps_pred_CCNN.shape[0],dtype=bool)].reshape(tps_pred_CCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"CNN","score":np.ndarray.flatten(np.ndarray.flatten(tps_pred_CNN[~np.eye(tps_pred_CNN.shape[0],dtype=bool)].reshape(tps_pred_CNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-CNN","score":np.ndarray.flatten(np.ndarray.flatten(tps_pred_PTCNN[~np.eye(tps_pred_PTCNN.shape[0],dtype=bool)].reshape(tps_pred_PTCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-GREEN","score":np.diagonal(tps_pred_CGREENSiSu),"mode":"WP"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"GREEN","score":np.diagonal(tps_pred_GREENSiSu),"mode":"WP"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-GREEN","score":np.diagonal(tps_pred_PTGREENSiSu),"mode":"WP"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"TS-LDA","score":np.diagonal(tps_pred_TSLDASiSu),"mode":"WP"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-CNN","score":np.diagonal(tps_pred_CCNNSiSu),"mode":"WP"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"CNN","score":np.diagonal(tps_pred_CNNSiSu),"mode":"WP"})],ignore_index=True)
df_tps_pred = pd.concat([df_tps_pred,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-CNN","score":np.diagonal(tps_pred_PTCNNSiSu),"mode":"WP"})],ignore_index=True)

df_recall = pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-GREEN","score":np.ndarray.flatten(recall_CGREEN[~np.eye(recall_CGREEN.shape[0],dtype=bool)].reshape(recall_CGREEN.shape[0],-1)),"mode":"DA"})
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"GREEN","score":np.ndarray.flatten(np.ndarray.flatten(recall_GREEN[~np.eye(recall_GREEN.shape[0],dtype=bool)].reshape(recall_GREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-GREEN","score":np.ndarray.flatten(np.ndarray.flatten(recall_PTGREEN[~np.eye(recall_PTGREEN.shape[0],dtype=bool)].reshape(recall_PTGREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"TS-LDA","score":np.ndarray.flatten(np.ndarray.flatten(recall_TSLDA[~np.eye(recall_TSLDA.shape[0],dtype=bool)].reshape(recall_TSLDA.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-CNN","score":np.ndarray.flatten(np.ndarray.flatten(recall_CCNN[~np.eye(recall_CCNN.shape[0],dtype=bool)].reshape(recall_CCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"CNN","score":np.ndarray.flatten(np.ndarray.flatten(recall_CNN[~np.eye(recall_CNN.shape[0],dtype=bool)].reshape(recall_CNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-CNN","score":np.ndarray.flatten(np.ndarray.flatten(recall_PTCNN[~np.eye(recall_PTCNN.shape[0],dtype=bool)].reshape(recall_PTCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-GREEN","score":np.diagonal(recall_CGREENSiSu),"mode":"WP"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"GREEN","score":np.diagonal(recall_GREENSiSu),"mode":"WP"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-GREEN","score":np.diagonal(recall_PTGREENSiSu),"mode":"WP"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"TS-LDA","score":np.diagonal(recall_TSLDASiSu),"mode":"WP"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-CNN","score":np.diagonal(recall_CCNNSiSu),"mode":"WP"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"CNN","score":np.diagonal(recall_CNNSiSu),"mode":"WP"})],ignore_index=True)
df_recall = pd.concat([df_recall,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-CNN","score":np.diagonal(recall_PTCNNSiSu),"mode":"WP"})],ignore_index=True)

df_f1 = pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-GREEN","score":np.ndarray.flatten(f1_CGREEN[~np.eye(f1_CGREEN.shape[0],dtype=bool)].reshape(f1_CGREEN.shape[0],-1)),"mode":"DA"})
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"GREEN","score":np.ndarray.flatten(np.ndarray.flatten(f1_GREEN[~np.eye(f1_GREEN.shape[0],dtype=bool)].reshape(f1_GREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-GREEN","score":np.ndarray.flatten(np.ndarray.flatten(f1_PTGREEN[~np.eye(f1_PTGREEN.shape[0],dtype=bool)].reshape(f1_PTGREEN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"TS-LDA","score":np.ndarray.flatten(np.ndarray.flatten(f1_TSLDA[~np.eye(f1_TSLDA.shape[0],dtype=bool)].reshape(f1_TSLDA.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"C-CNN","score":np.ndarray.flatten(np.ndarray.flatten(f1_CCNN[~np.eye(f1_CCNN.shape[0],dtype=bool)].reshape(f1_CCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"CNN","score":np.ndarray.flatten(np.ndarray.flatten(f1_CNN[~np.eye(f1_CNN.shape[0],dtype=bool)].reshape(f1_CNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.repeat(np.linspace(1,24,24),23),"pipeline":"PS-CNN","score":np.ndarray.flatten(np.ndarray.flatten(f1_PTCNN[~np.eye(f1_PTCNN.shape[0],dtype=bool)].reshape(f1_PTCNN.shape[0],-1))),"mode":"DA"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-GREEN","score":np.diagonal(f1_CGREENSiSu),"mode":"WP"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"GREEN","score":np.diagonal(f1_GREENSiSu),"mode":"WP"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-GREEN","score":np.diagonal(f1_PTGREENSiSu),"mode":"WP"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"TS-LDA","score":np.diagonal(f1_TSLDASiSu),"mode":"WP"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"C-CNN","score":np.diagonal(f1_CCNNSiSu),"mode":"WP"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"CNN","score":np.diagonal(f1_CNNSiSu),"mode":"WP"})],ignore_index=True)
df_f1 = pd.concat([df_f1,pd.DataFrame({"dataset":"RickerBCVEP","subject":np.linspace(1,24,24),"pipeline":"PS-CNN","score":np.diagonal(f1_PTCNNSiSu),"mode":"WP"})],ignore_index=True)


names = ["C-GREEN","GREEN","PS-GREEN","TS-LDA","C-CNN","CNN","PS-CNN"]


#############################

def raincloud_graph(results,title,xlabel,xlim=None):
    rcParams['font.weight'] = 'bold'

    x = "mode"
    y = "score"
    hue = "pipeline"

    boxplot_lw = 1.0
    boxplot_props = {'linewidth': boxplot_lw}

    temp = results.copy()
    # temp.insert(1,"procedure",np.repeat(["DG","DA","SS","DG","DA","SS"],12*1),True)

    n_pi = 3
    palette = dict(zip(names*579, sns.color_palette('Set2', len(names))*579))
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(6)
    fig.set_figwidth(8)
    pt.RainCloud(
        data=results, y=y, x=x,
        hue=hue,
        bw='scott',
        width_viol=0, width_box=0.8, point_size=5,
        dodge=True, orient='h',
        linewidth=0, box_linewidth=boxplot_lw,
        box_whiskerprops=boxplot_props,
        box_medianprops=boxplot_props,
        alpha=0.7, palette=palette,
        box_showfliers=False,
        fontsize=17,
        ax=ax, pointplot=True,
        point_linestyles="none",
        point_markers='D',)



    # ax.set_yticks([])

    for i in range(n_pi):
        ax.axhline(y=i + 0.5, xmin=0, xmax=1.02, color='black' , linestyle=':', alpha=0.3)
    if xlim is not None:
        ax.set_xlim(left=0,right=xlim)
    # GREEN_patch = mpatches.Patch(color=sns.color_palette('Set2', 5)[0], label='Domain Generalization')
    # PTGREEN_patch = mpatches.Patch(color=sns.color_palette('Set2', 5)[1], label='Domain Adaptation')
    # SS_patch = mpatches.Patch(color=sns.color_palette('Set2', 5)[2], label='Single Subject')
    # SS_patch = mpatches.Patch(color=sns.color_palette('Set2', 5)[3], label='Single Subject')
    # SS_patch = mpatches.Patch(color=sns.color_palette('Set2', 5)[4], label='Single Subject')

    ax.set_xlabel(xlabel,fontsize=17, fontweight='bold')
    ax.set_ylabel("mode",fontsize=17, fontweight='bold')
    ax.set_title(title,fontsize=20, fontweight='bold')
    # ax.legend(['Domain Generalization', 'Domain Adaptation', 'Single Subject'],
    #         handles=[DG_patch,DA_patch,SS_patch],
    #             loc=(0.05, 1.01), ncols=n_pi)

raincloud_graph(df_score,"Balanced epoch-level accuracy","Balanced epoch-level accuracy",1.02)
plt.save("../figures/RickerBCVEP_epoch_level_accuracy.png",dpi=300)
raincloud_graph(df_score_code,"trial-level accuracy","Trial-level accuracy",1.02)
plt.save("../figures/RickerBCVEP_trial_level_accuracy.png",dpi=300)
raincloud_graph(df_tps_train,"Training time","Time in seconds")
plt.save("../figures/RickerBCVEP_training_time.png",dpi=300)
raincloud_graph(df_tps_pred,"Test set prediction time","Time in seconds",50)
plt.save("../figures/RickerBCVEP_prediction_time.png",dpi=300)
raincloud_graph(df_f1,"f1 score","score",1.02)
plt.save("../figures/RickerBCVEP_f1_score.png",dpi=300)
raincloud_graph(df_recall,"Recall score","score",1.02)
plt.save("../figures/RickerBCVEP_recall_score.png",dpi=300)



####################  Correlation measures ##########################

measure_1point = pd.read_pickle("../complements/1point_results.pkl").replace(np.NaN, None)
index_order = pd.read_pickle("../complements/index_order.pkl").replace(np.NaN, None)
results = pd.read_pickle("../complements/results.pkl").replace(np.NaN, None)


metric = "score"
algos = ["PTGREEN"]
markers = [".","x"]
palette = sns.color_palette('Set2', 4)



fig, ax1 = plt.subplots(1)
fig.set_figheight(6)
fig.set_figwidth(8)
for i in range(len(algos)):
    # Only alphas and deltas are plotted and only AT in SiSu
    algo = algos[i]
    lin_reg = linear_model.LinearRegression()
    X = results["SiSu"][metric][algo][index_order["SiSu"][metric][algo]]
    yalphas = measure_1point["alphas"]["AT"][index_order["SiSu"][metric][algo]]
    ydeltas = measure_1point["deltas"]["AT"][index_order["SiSu"][metric][algo]] 
    slopedeltas,interdeltas,r_valuedeltas,pdeltas,std_errdeltas = stats.linregress(X,ydeltas)
    slopealphas,interalphas,r_valuealphas,palphas,std_erralphas = stats.linregress(X,yalphas)
    X = X.reshape(-1,1)
    modeldeltas = lin_reg.fit(X,ydeltas)
    responsedeltas = modeldeltas.predict(X)

    modelalphas = lin_reg.fit(X,yalphas)
    responsealphas = modelalphas.predict(X)

    color = palette[0]
    ax1.set_xlabel('WP balanced epoch-level accuracy', fontsize=15, fontweight="bold")
    ax1.set_ylabel(u'δ bandpower', color=color, fontsize=15, fontweight="bold")
    ax1.scatter(X,ydeltas,color=color,label="deltas",alpha=0.7, marker=markers[i],s=66)
    ax1.plot(X,responsedeltas,color=color,linewidth=4)
    dy = responsedeltas[1] - responsedeltas[0]
    dx = X[1] - X[0]
    angle = np.rad2deg(np.arctan2(dy, dx))[0]
    ptext = f"p = {np.round(pdeltas,decimals=3)}" if pdeltas>0.001 else "p < 0.001"
    # annotate with transform_rotates_text to align text and line
    ax1.text(X[1], responsedeltas[1], f'R={r_valuedeltas:.2f} | '+ptext, ha='left', va='bottom',
            transform_rotates_text=True, rotation=angle, rotation_mode='anchor', color=color, fontsize=14, fontweight="bold")
    ax1.tick_params(axis='y', labelcolor=color, labelsize=13)
    ax1.set_yticklabels(ax1.get_yticklabels(), weight='bold')

    ax1.tick_params(axis='x', labelsize=13)
    ax1.set_xticklabels(ax1.get_xticklabels(), weight='bold')

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = palette[1]
    ax2.set_ylabel('α bandpower', color=color, fontsize=15, fontweight="bold")
    ax2.scatter(X,yalphas,color=color,label="alphas",alpha=0.7,marker=markers[i],s=66)
    ax2.plot(X,responsealphas,color=color,linewidth=4)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=13)
    ax2.set_yticklabels(ax2.get_yticklabels(), weight='bold')
    dy = responsealphas[1] - responsealphas[0]
    dx = X[1] - X[0]
    angle = np.rad2deg(np.arctan2(dy, dx))[0]
    # annotate with transform_rotates_text to align text and line
    ptext = f"p = {np.round(palphas,decimals=3)}" if palphas>0.001 else "p < 0.001"
    ax2.text(X[1], responsealphas[1], f'R={r_valuealphas:.2f} | '+ptext, ha='left', va='bottom',
            transform_rotates_text=True, rotation=angle, rotation_mode='anchor', color=color, fontsize=14, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor=color, labelsize=13)
    ax2.set_yticklabels(ax2.get_yticklabels(), weight='bold')
#     plt.xlim((0.5,1.0))
plt.title("Spectral neurophysiological predictors", fontsize=18)
plt.savefig("../figures/RickerBCVEP_spectral_predictors.png",dpi=300)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

fig2, ax3 = plt.subplots()
fig2.set_figheight(6)
fig2.set_figwidth(8)
for i in range(len(algos)):
    # Only amp and corr are plotted and only AT in SiSu
    algo = algos[i]
    lin_reg = linear_model.LinearRegression()
    X = results["SiSu"][metric][algo][index_order["SiSu"][metric][algo]]
    yamp = measure_1point["amp"]["AT"][index_order["SiSu"][metric][algo]]/np.sum(measure_1point["amp"]["AT"][index_order["SiSu"][metric][algo]])
    ycorr = measure_1point["corr_mean"]["AT"][index_order["SiSu"][metric][algo]]
    slopecorr,intercorr,r_valuecorr,pcorr,std_errcorr = stats.linregress(X,ycorr)
    slopeamp,interamp,r_valueamp,pamp,std_erramp = stats.linregress(X,yamp)
    X = X.reshape(-1,1)
    modelcorr = lin_reg.fit(X,ycorr)
    responsecorr = modelcorr.predict(X)

    modelamp = lin_reg.fit(X,yamp)
    responseamp = modelamp.predict(X)

    color = palette[0]
    ax3.set_xlabel('WP balanced epoch-level accuracy', fontsize=15, fontweight="bold")
    ax3.set_ylabel('correlation between epochs', color=color, fontsize=15, fontweight="bold")
    ax3.scatter(X,ycorr,color=color,label="corr",alpha=0.7, marker=markers[i],s=66)
    ax3.plot(X,responsecorr,color=color,linewidth=4)
    dy = responsecorr[1] - responsecorr[0]
    dx = X[1] - X[0]
    angle = np.rad2deg(np.arctan2(dy, dx))[0]
    ptext = f"p = {np.round(pcorr,decimals=3)}" if pcorr>0.001 else "p < 0.001"
    # annotate with transform_rotates_text to align text and line
    ax3.text(X[1], responsecorr[1], f'R={r_valuecorr:.2f} | '+ptext, ha='left', va='bottom',
            transform_rotates_text=True, rotation=angle, rotation_mode='anchor', color=color, fontsize=14, fontweight="bold")
    ax3.tick_params(axis='y', labelcolor=color, labelsize=13)
    ax3.tick_params(axis='x', labelsize=13)
    ax3.set_yticklabels(ax3.get_yticklabels(), weight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), weight='bold')

    ax4 = ax3.twinx()  # instantiate a second Axes that shares the same x-axis

    color = palette[1]
    ax4.set_ylabel('Peak-to-peak amplitude', color=color, fontsize=15, fontweight="bold")
    ax4.scatter(X,yamp,color=color,label="amp",alpha=0.7,marker=markers[i],s=66)
    ax4.plot(X,responseamp,color=color,linewidth=4)
    ax4.tick_params(axis='y', labelcolor=color, labelsize=13)
    dy = responseamp[1] - responseamp[0]
    dx = X[1] - X[0]
    angle = np.rad2deg(np.arctan2(dy, dx))[0]
    # annotate with transform_rotates_text to align text and line
    ptext = f"p = {np.round(pamp,decimals=3)}" if pamp>0.001 else "p < 0.001"
    ax4.text(X[1], responseamp[1], f'R={r_valueamp:.2f} | '+ptext, ha='left', va='bottom',
            transform_rotates_text=True, rotation=angle, rotation_mode='anchor', color=color, fontsize=14, fontweight="bold")
    ax4.tick_params(axis='y', labelcolor=color, labelsize=13)
    ax4.set_yticklabels(ax4.get_yticklabels(), weight='bold')
#     plt.xlim((0.5,1.0))

plt.title("ERP-based neurophysiological predictors", fontsize=18)
fig2.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("../figures/RickerBCVEP_ERP_predictors.png",dpi=300)

# plt.xlim((0.5,1.0))
plt.show()


measure_1point["Correlation between epochs"] = measure_1point["corr_mean"]
measure_1point["Peak-to-peak amplitude"] = measure_1point["amp"]
del(measure_1point["corr_mean"])
del(measure_1point["amp"])

rcParams['font.weight'] = 'bold'
keys = measure_1point.keys()
diff_point = {}
for i,k in enumerate(keys):
    diff_point[k] = {}
    if k in ["SNR_mean","SNR_std"]:
        diff_point[k]["APWave/AP difference"] = None
        diff_point[k]["AP/BP difference"] = None
    elif k in ["alphas","betas","deltas","thetas"]:
        diff_point[k]["APWave/AP difference"] = None
        diff_point[k]["AP/BP difference"] = measure_1point[k]["AT"] - measure_1point[k]["BT"]
    else:
        if k!='amp':
            diff_point[k]["APWave/AP difference"] = np.mean([measure_1point[k]["Wave_BT"] - measure_1point[k]["BT"],measure_1point[k]["Wave_AT"] - measure_1point[k]["AT"]],axis=0)
            diff_point[k]["AP/BP difference"] = np.mean([measure_1point[k]["AT"] - measure_1point[k]["BT"],measure_1point[k]["Wave_AT"] - measure_1point[k]["Wave_BT"]],axis=0)
        else:
            diff_point[k]["APWave/AP difference"] = np.mean([measure_1point[k]["Wave_BT"]/np.sum(measure_1point[k]["Wave_BT"]) - measure_1point[k]["BT"]/np.sum(measure_1point[k]["BT"]),measure_1point[k]["Wave_AT"]/np.sum(measure_1point[k]["Wave_AT"]) - measure_1point[k]["AT"]/np.sum(measure_1point[k]["AT"])],axis=0)
            diff_point[k]["AP/BP difference"] = np.mean([measure_1point[k]["AT"]/np.sum(measure_1point[k]["AT"]) - measure_1point[k]["BT"]/np.sum(measure_1point[k]["BT"]),measure_1point[k]["Wave_AT"]/np.sum(measure_1point[k]["Wave_AT"]) - measure_1point[k]["Wave_BT"]/np.sum(measure_1point[k]["Wave_BT"])],axis=0)


X = np.linspace(1,24,24)
save = False

metric = "score"
algo = "PTGREEN"


for i,k in enumerate(keys):
    fig, ax1 = plt.subplots(1)
    fig.set_figheight(6)
    fig.set_figwidth(8)
    palette = dict(zip(["a","b","c",'APWave/AP difference','AP/BP difference'], sns.color_palette('Set2', 8)))
    # ax1 = plt.subplot(1,2,1)
    for j in ["APWave/AP difference","AP/BP difference"]:
        lin_reg = linear_model.LinearRegression()
        if diff_point[k][j] is None:
            continue
        else:
            X = results["SiSu"][metric][algo][index_order["SiSu"][metric][algo]]
            y = diff_point[k][j][index_order["SiSu"][metric][algo]]
            slope,inter,r_value,p,std_err = stats.linregress(X,y)
            X = X.reshape(-1,1)
            model = lin_reg.fit(X,y)
            response = model.predict(X)

            # ptext = f"p = {np.round(p,decimals=3)}" if p>0.001 else "p < 0.001"

            dy = response[1] - response[0]
            dx = X[1] - X[0]
            angle = np.rad2deg(np.arctan2(dy, dx))[0]
            # annotate with transform_rotates_text to align text and line
            ptext = f"p = {np.round(p,decimals=3)}" if p>0.001 else "p < 0.001"
            ax1.text(X[1], response[1], f'R={r_value:.2f} | '+ptext, ha='left', va='bottom',
                    transform_rotates_text=True, rotation=angle, rotation_mode='anchor', color=palette[j], fontsize=14, fontweight='bold')
            # plt.scatter(X,y,color=palette[j],label=f"{j} and R = {np.round(r_value,decimals=2)} | "+ptext,alpha=0.7)
            plt.scatter(X,y,color=palette[j],label=f"{j}",alpha=0.7,marker='.',s=66)
            plt.plot(X,response,color=palette[j],linewidth=4)
        # plt.xticks(results["SiSu"][metric][algo][index_order["SiSu"][metric][algo]],index_order["SiSu"][metric][algo]+1)
    plt.title(k, fontsize=18)
    # plt.title("WP "+k, fontsize=18)
    plt.xlabel("WP balanced epoch-level accuracy", fontsize=15, weight='bold')
    plt.ylabel(k, fontsize=15, weight='bold')
    plt.tick_params(axis="both", labelsize=13)
    plt.xlim((0.5,1.0))
    plt.legend()
    

    plt.legend(fancybox=True, framealpha=0.2)

    plt.savefig(f"../figures/Space_diff_score_ordered_stats_{algo}_{k}.png")