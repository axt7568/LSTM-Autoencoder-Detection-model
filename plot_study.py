#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Arjun Thangaraju'
# ---------------------------------------------------------------------------
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

with open('C:/Users/arjun.thangaraju/PycharmProjects/Autoencoder_Anomaly_Detection/lstm_autoencoder_anomaly_detection/ROC_Saved_Values/4f_abs_mod.pickle', 'rb') as f:
    [auc_values, scored, loss_data] = pickle.load(f)

# plot the loss distribution of the test set
plt.figure(figsize=(16, 9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins=20, kde=True, color='blue')
plt.xlim([0.0, .5])
plt.show()

# plot bearing failure time plot
index = np.where(auc_values == auc_values.max())[0][0]
date_index = scored.axes[0].tolist()
scored['Threshold'] = loss_data[index]
print('Threshold value = %.5f' %loss_data[index])
scored.plot(logy=True, figsize=(16, 16), ylim=[1e-2, 1e2], color=['blue', 'red'])
plt.axvline(x=index, color='g')
plt.title(' Loss vs Threshold with Threshold value as %.5f' %loss_data[index])
print('Max AUC = %3f occurs at sample number %d' % (np.max(auc_values), np.where( auc_values == auc_values.max())[0][0]))
print('Max AUC = %3f occurs at date: '% np.max(auc_values),  date_index[index])
plt.show()
