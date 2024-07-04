import neurokit2 as nk
import numpy as np
import pandas as pd


df_train_features=pd.read_csv('X_train.csv')
Y=pd.read_csv('y_train.csv')
#x_test_f=pd.read_csv('X_test.csv')

def Vector(feature,label):
    label=label.iloc[:,1]
    feature=feature.iloc[:,1:]
    #X_train, X_test, y_train, y_test  = train_test_split(feature,label, test_size=0.2, random_state=222)
    return feature,label


X_values,Y_labels= Vector(df_train_features,Y)

X_features, info = nk.ecg_process(X_values, sampling_rate=300)

df = pd.DataFrame(X_features)
df.to_csv('neurokit_features.csv',index=False)

print(info)