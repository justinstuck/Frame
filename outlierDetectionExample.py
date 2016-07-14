# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:27:22 2016

@author: Justin.Stuck
"""

print(__doc__)
import pandas as pd
from StringIO import StringIO
from requests import get
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy import stats

from sklearn import svm
from sklearn.covariance import EllipticEnvelope

from sklearn import preprocessing


def get_data():
    uri = 'http://ds/api/warehouse/all-the-frame-metrics-by-step?fmt=csv'
    resp = get(uri)
    history = pd.read_csv(StringIO(resp.text))
    study = '4BC154FD-6429-444E-B589-4B3B7CF1BDA6'
    #'4BC154FD-6429-444E-B589-4B3B7CF1BDA6'
    dqs = pd.read_csv("C:/Users/Justin.Stuck/Desktop/JDQs.csv",low_memory=False)['ticketId']
    history = pd.concat([history[history.name =='Unity Frame Shopping'],history[history.name =='Unity Frame Tutorial']])
    history['latency'] = preprocessing.scale(history['latency'].apply(lambda x: math.log(x+.0000000000001)))
    history['bandwidth'] = preprocessing.scale(history['bandwidth'].apply(lambda x: math.log(x+.0000000000001)))
    history['framerate'] = preprocessing.scale(history['framerate'])
    dqs = history[history['ticketid'].isin(dqs)]   
    history = history[history['studyidguid']==str.lower(study)]
    history = history[history.iscomplete == 1]
    
    history = pd.concat([history,dqs])
              
    #respondents = history[history['studyidguid']==studyid.lower()]
    return history[history.name == 'Unity Frame Shopping'], dqs[dqs.name== 'Unity Frame Shopping']
    
        


# Example settings
data, dqs = get_data()
n_samples = data.shape[0]
print n_samples

outliers_fraction = 0.10
clusters_separation = [0]

# define two outlier detection tools to be compared
classifiers = {
    "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                     kernel="rbf", gamma=0.1),
    "robust covariance estimator": EllipticEnvelope(contamination=.1)}


xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-3, 3, 500))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.ones(n_samples, dtype=int)
ground_truth[-n_outliers:] = 0

X = zip(data['bandwidth'],data['latency'])
#plt.scatter(X[:,0],X[:,1])
#print X
plt.show()
plt.figure(figsize=(10, 5))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    # fit the data and tag outliers
    clf.fit(zip(data['bandwidth'],data['latency']))
    data['y_pred'] = clf.decision_function(X).ravel()
    threshold = stats.scoreatpercentile(data['y_pred'],
                                        100 * outliers_fraction)
    data['y_pred'] = data['y_pred'] > threshold
    n_errors = (data['y_pred'] != ground_truth).sum()
    # plot the levels lines and the points
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    subplot = plt.subplot(1, 2, i + 1)
    subplot.set_title("Outlier detection")
    subplot.set_xlabel('bandwidth')
    subplot.set_ylabel('latency')
    
    subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                     cmap=plt.cm.Blues_r)
    cs = subplot.contour(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
    plt.clabel(cs, colors='k', fontsize=14)
    
    a = subplot.contour(xx, yy, Z, levels=[threshold],
                        linewidths=2, colors='red')
    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                     colors='orange')
    b = subplot.scatter(data['bandwidth'],data['latency'], c='white')
    c = subplot.scatter(dqs['bandwidth'],dqs['latency'], c='black')
    subplot.axis('tight')
    subplot.legend(
        [a.collections[0], b''', c'''],
        ['learned decision function', 'true inliers', 'true outliers'],
        prop=matplotlib.font_manager.FontProperties(size=11))
    #subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
    subplot.set_xlim((-5, 5))
    subplot.set_ylim((-3, 3))
plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)

plt.show()
