import tsfel
import pandas as pd
import tqdm
import numpy as np


X_train_orig = pd.read_csv('X_test.csv')
feature=X_train_orig.iloc[:,1:2459].dropna(axis=1).to_numpy()


# Retrieves a pre-defined feature configuration file to extract all available features
cfg = tsfel.get_features_by_domain()

# Extract features
print(feature.shape)
X = tsfel.time_series_features_extractor(cfg, feature[0], fs=300, verbose=1)
features = np.zeros((feature.shape[0], X.shape[1]))
for i in tqdm.tqdm(range(feature.shape[0])):
    features[i] = tsfel.time_series_features_extractor(cfg, feature[i], fs=300, verbose=0)
X=X.to_numpy()

df = pd.DataFrame(features)
df.to_csv('tsfel_features_test.csv')