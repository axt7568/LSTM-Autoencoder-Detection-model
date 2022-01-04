import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from statistics import mean
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import time

# Start Timer
start_time = time.time()

with open('C:/Users/arjun.thangaraju/PycharmProjects/LSTM_Autoencoder_Anomaly_Detection/Saved_Pickle_Data_Files/12f_abs_sd_bp_mod.pickle', 'rb') as f:
    [merged_data, train, test, X_train, X_test] = pickle.load(f)

# Load h5 model
model = load_model('C:/Users/arjun.thangaraju/PycharmProjects/LSTM_Autoencoder_Anomaly_Detection/Saved_h5_models/12f_abs_sd_bp_mod.h5')

model.summary()

# calculate the loss on the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index
scored_train = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored_train['Loss_mae'] = np.mean(np.abs(X_pred - Xtrain), axis=1)

# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index
scored_test = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored_test['Loss_mae'] = np.mean(np.abs(X_pred - Xtest), axis=1)

scored = pd.concat([scored_train, scored_test])
# scored = scored_train
scored['Threshold'] = 1
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
loss_data = scored['Loss_mae'].to_numpy()
date_values = scored.index.to_numpy()
# trueLabel = np.copy(loss_data)
trueLabel = np.zeros(len(loss_data))
max_auc_each_date = np.zeros(len(loss_data))
threshold_values = np.zeros(len(loss_data))
auc_values = np.zeros(len(loss_data))

for i in range ((int)(len(loss_data)*0.3),(int)(len(loss_data)*0.6)): # For each date
    conjecture_failure_on_set_time = i
    trueLabel[:conjecture_failure_on_set_time] = 0
    trueLabel[conjecture_failure_on_set_time:] = 1
    # Calculate roc curve
    fpr, tpr, thresholds = roc_curve(trueLabel, loss_data)
    # auc = auc(fpr, tpr)
    # Calculate AUC
    auc = roc_auc_score(trueLabel, loss_data)
    auc_values[i] = auc
print('Max AUC = %3f occurs at sample number %d' % (np.max(auc_values), np.where( auc_values == auc_values.max())[0][0]))
print("--- Elapsed Time is: %s seconds ---" % (time.time() - start_time))

with open('C:/Users/arjun.thangaraju/PycharmProjects/LSTM_Autoencoder_Anomaly_Detection/ROC_Saved_Values/12f_abs_sd_bp_mod.pickle', 'wb') as f:
    pickle.dump([auc_values, scored, loss_data], f)

# Plot AUC values
plt.plot(auc_values, 'r')
plt.show()
