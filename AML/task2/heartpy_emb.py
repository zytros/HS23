import numpy as np
import pandas as pd
import heartpy as hp

df_train_features=pd.read_csv('X_test.csv')
Y=pd.read_csv('y_train.csv')
#x_test_f=pd.read_csv('X_test.csv')

def Vector(feature,label):
    label=label.iloc[:,1]
    feature=feature.iloc[:,1:]
    #X_train, X_test, y_train, y_test  = train_test_split(feature,label, test_size=0.2, random_state=222)
    return feature,label

def Extraction(X_train):
    results_all= np.zeros((np.size(X_train,axis=0),19))
    for i in range(np.size(X_train,axis=0)):
        feature=X_train.iloc[i,:]
        feature=feature.dropna(axis=0)
        signal1=np.array(feature)
        signal1=signal1.flatten()
        #peaks=ecg.ecg(signal=signal1, sampling_rate=300.0, show=False)
        #t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(feature, show=False)[:3]
        #nni = tools.nn_intervals(t[rpeaks])
        #analyze_df = hrv(rpeaks=t[rpeaks], sampling_rate=300, plot_ecg=False,plot_tachogram=False, show=False)
        wd, m = hp.process(signal1, sample_rate = 300.0, bpmmin = 40, bpmmax = 120)
        z=0
        for measure in m.keys():
                results_all[i,z] = m[measure]
                z=z+1
                if z==12:
                    results_all[i,13] = np.amax(signal1)
                    results_all[i,14] = np.amin(signal1)
                    results_all[i,15] = np.median(signal1)
                    results_all[i,16] = np.mean(signal1)
                    results_all[i,17] = np.std(signal1)
                    results_all[i,18] = np.var(signal1)
        if i==3500:
            print(3500)
            print(m.keys())
    return results_all

X_values,Y_labels= Vector(df_train_features,Y)

X_features=Extraction(X_values)

df = pd.DataFrame(X_features)
df.to_csv('heartpy_features_test.csv',index=False)