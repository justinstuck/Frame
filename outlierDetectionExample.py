# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:27:22 2016

@author: Justin.Stuck
"""

#print(__doc__)
import pandas as pd
from StringIO import StringIO
from requests import get
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy import stats
from scipy.stats import chi2

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing


def chi2_outliers(data, confidences,df):
    #returns list of lists of respondents removed at confidence levels given by confidences
        #Mahalanobis distances define multidimensional ellipsoids. The square of the distances follow a chi-square distribution with p degrees of freedom
        #where p is the number of random variables in the multivariate distribution. Ref Warre, Smith, Cybenko "Use of Mahalanobis Distance..." and Johnson and Wichern (2007, p. 155, Eq. 4-8)    
    return [data[data['MDist']>= chi2.ppf(conf,df)] for conf in confidences] 
    
    
    #x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), 100)
    #print "chi-square 95% quantile for {} degrees of freedom is {}".format(df,chi2.ppf(.95,df))
    
        #plt.plot(x, chi2.pdf(x, df),'r-', lw=5, alpha=0.6, label='chi2 pdf')

def transform_and_scale(data, transform_vars,other_vars):
    for u in transform_vars:
        data[u] = preprocessing.scale(data[u].apply(lambda x: math.log(x+.0000000000000001)))
    for v in other_vars:
        data[v] = preprocessing.scale(data[v])
    return data
        
class FrameCleaner():
    def __init__(self,studyid):
        self.studyid = studyid
        self.features = [ 'bandwidth', 'latency', 'framerate']
        self.experiences = ['Shopping', 'Tutorial']
    

    def get_data(self):
        uri = 'http://ds/api/warehouse/all-the-frame-metrics-by-step?fmt=csv'
        resp = get(uri)
        history = pd.read_csv(StringIO(resp.text))
        #study = studyid'4BC154FD-6429-444E-B589-4B3B7CF1BDA6'
        dqs = pd.read_csv("C:/Users/Justin.Stuck/Desktop/JDQs.csv",low_memory=False)['ticketId']
        history = history[history.name =='Unity Frame Shopping']
        history = history[history.iscomplete == 1]
        rawShopHistory = history.copy(deep=True)
    
        history = transform_and_scale(history,['latency','bandwidth'],['framerate'])
        
        dqs = history[history['ticketid'].isin(dqs)]   
        history = history[history['studyidguid']==str.lower(self.studyid)]
    
        
        history = pd.concat([history,dqs])
                  
        #respondents = history[history['studyidguid']==studyid.lower()]
        print rawShopHistory
        return history, dqs[dqs.name== 'Unity Frame Shopping'], rawShopHistory#, rawShopHistory[rawShopHistory['studyidguid']==str.lower(study) & rawShopHistory['iscomplete']==1] 
        
    
    def calc(self,outliers_fraction):
        

        data, dqs, raw = self.get_data()        
        clf = EllipticEnvelope(contamination=outliers_fraction)
        X = zip(data['bandwidth'],data['latency'],data['framerate'])
        clf.fit(X)        
        #data['y_pred'] = clf.decision_function(X).ravel()
        data['y_pred'] = clf.decision_function(X).ravel()
        
        threshold = np.percentile(data['y_pred'],
                                            100 * outliers_fraction)
        data['MDist']=clf.mahalanobis(X)
        
        #picking "bad" outliers, not good ones
        outliers = chi2_outliers(data, [.8,.9,.95], 3)
        outliers = [i[i['bandwidth']<i['latency']] for i in outliers]
        
        #outliers = data[data['y_pred']<threshold]
        #data['y_pred'] = data['y_pred'] > threshold
        #outliers = [x[['ticketid','MDist']].merge(raw, how='inner').drop_duplicates() for x in outliers]
        #print raw
        outliers = [raw[raw['ticketid'].isin(j['ticketid'])] for j in outliers]
        outliers = [k[k['framerate']<(k['framerate'].mean()+k['framerate'].std())] for k in outliers] #making sure we don't remove aberrantly good framrates
        #outliers.sort_values(by='MDist',inplace=True)
        dqs = raw[raw['ticketid'].isin(dqs['ticketid'])]
        return outliers, dqs
    
    def to_excel(data,sheetnames,filename):
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        for data, sheet in zip(data,sheetnames):
            data.to_excel(writer, sheet_name=sheet)
        writer.save()
#plot3d()    
fc = FrameCleaner('4BC154FD-6429-444E-B589-4B3B7CF1BDA6')
outliers,dqs = fc.calc(outliers_fraction = 0.10)
#outliers['Removed by Insights'] = outliers['ticketid'].isin(dqs['ticketid'])
#extraInsights = dqs[~(dqs['ticketid'].isin(outliers['ticketid']))]

#data['Removed by insights']  
#to_excel(outliers,extraInsights,'mahalTest.xlsx')











        
        
        
        
        
        
        
        

        
        #n_samples = data.shape[0]
        #n_errors = (data['y_pred'] != ground_truth).sum()
'''
        bins = np.linspace(data['MDist'].min(),data['MDist'].max(),30)
        plt.hist(data['MDist'], bins, alpha=0.8, label='Mahalanobis Distance',color='green')    
        plt.title("Frame Mahalanobis Distance Distribution", weight='bold',size=15)
        plt.xlabel("Squared Mahalanobis Distance",size=14)

        
    
        
        # plot the levels lines and the points
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        plt.title("Outlier Detection for Jarden Study 4BC154FD-6429-444E-B589-4B3B7CF1BDA6",weight='bold')
        plt.xlabel('bandwidth')
        plt.ylabel('latency')    
        
        
        
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
        cs = plt.contour(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
        plt.clabel(cs, colors='k', fontsize=14)
        
        a = plt.contour(xx, yy, Z, levels=[threshold],
                            linewidths=2, colors='red')
        plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
        b = plt.scatter(data['bandwidth'],data['latency'], c='white')
        c = plt.scatter(dqs['bandwidth'],dqs['latency'], c='black')
         
        
        
        
    
        outies = plt.scatter(outliers['bandwidth'],outliers['latency'], s=100, facecolors='none', edgecolors='g')
        plt.axis('tight')
        plt.legend(
            [a.collections[0], b, c, outies],
            ['learned decision function', 'true inliers', 'true outliers', 'Recommended for Removal'],
            prop=matplotlib.font_manager.FontProperties(size=11))
        #subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        plt.xlim((-5, 5))
        plt.ylim((-3, 3))
        #plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
        
        plt.show()
        
    
    def plot3d():
        outliers_fraction = .1
        data, dqs = get_data()
        n_samples = data.shape[0]
        clf = EllipticEnvelope(contamination=.1)
        
        #xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-3, 3, 500))
        n_inliers = int((1. - outliers_fraction) * n_samples)
        n_outliers = int(outliers_fraction * n_samples)
        ground_truth = np.ones(n_samples, dtype=int)
        ground_truth[-n_outliers:] = 0
        
        
        
        
        X = zip(data['bandwidth'],data['latency'],data['framerate'])
        
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        clf.fit(zip(data['bandwidth'],data['latency'],data['framerate']))
        xx, yy = np.meshgrid(np.linspace(data['bandwidth'].min(), data['bandwidth'].max(), 500), np.linspace(data['latency'].min(), data['latency'].max(), 500))
        zz = yy-xx
        ax.scatter(data['bandwidth'],data['latency'],data['framerate'], c='b', marker='o')
        ax.scatter(dqs['bandwidth'],dqs['latency'],dqs['framerate'], c='r', marker='^')
        ax.hold(True)
    
        ax.plot_surface(xx,yy,zz,color='orange',linewidth=0)
        ax.set_xlabel('bandwidth')
        ax.set_ylabel('latency')
        ax.set_zlabel('framerate')
        ax.view_init(elev=0., azim=90) 
        plt.show()
      
    def to_excel(data,extras,filename):
        
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        data.to_excel(writer, sheet_name='Removal Candidates')
        extras.to_excel(writer, sheet_name='Not Removed by Classifier')
        writer.save()
'''






