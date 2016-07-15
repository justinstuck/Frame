from StringIO import StringIO
from requests import get
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import statsmodels.api as sm
import itertools
import math
import scipy.stats as stats
from cStringIO import StringIO

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


class FrameCleaner():
    def __init__(self,studyid):
        self.studyid = studyid
        self.features = [ 'bandwidth', 'latency', 'framerate']
        self.experiences = ['Shopping', 'Tutorial']
        self.charts = StringIO()
    def Normalize(self, data):
        features = self.features
        sc = StandardScaler()
        normData = {x:sc.fit_transform(np.array(data[x])) for x in features}
        return pd.DataFrame(normData,index=data.ticketid)

    def calc_stats(self, data):
        features = self.features
        d = {'mean' : pd.Series([data[x].mean() for x in features], index=features),
             'std' : pd.Series([data[x].std() for x in features], index=features)}
        for i in range(1,4):
            d[('Q'+str(i))] = pd.Series([np.percentile(data[x],25*i) for x in features], index=features)

        self.stats = pd.DataFrame(d,index=features).transpose()
        return self.stats
        
    def plot_histos(self,features):
        if features != None:
            feats = features
        else:
            feats = self.features
        fig = plt.figure()
        fig = plt.gcf()
        fig.set_size_inches(10.5, 10.5)
        #fig.savefig('test2png.png', dpi=100)
        for feature in feats:
            datashop =self.shopping[feature]
            datatut = self.tutorial[feature]
            if feature == 'framerate':
                bins = np.linspace(datashop[datashop>0].min(),datashop[datashop>0].max(),26)
            else:
                bins = np.linspace(datashop[datashop>0].min(),datashop[datashop>0].max(),40)
                #bins = np.linspace(datashop[datashop>0].min(),200,50)
            plt.subplot(len(feats), 1, feats.index(feature)+1)
            plt.hist(datashop, bins, alpha=0.8, label='Shopping')
            plt.hist(datatut, bins, alpha=0.8, label='Tutorial', color='crimson')
            stats = self.stats
            ymin, ymax = plt.ylim()
            for i in range(1,4):
                quartile = ('Q'+str(i))
                plt.axvline(x=stats[feature][quartile],linewidth=3,color='g')
                plt.text(stats[feature][quartile],ymax/2,quartile+' = {0:.2f}'.format(stats[feature][quartile]),rotation=-90,color='k',weight = 'bold')

            #fit_alpha, fit_loc, fit_beta=stats.gamma.fit(data.shopping)
            plt.legend(loc='upper left', frameon=False)
            plt.xlabel(feature)
        plt.tight_layout()
        fig.savefig(self.charts)
    def transform(self, data, features):
        for feat in features:
            data[feat] = data[feat].apply(lambda x: math.log(x+.0000000000001))
        return data
        
    def get_data(self):
        uri = 'http://ds/api/warehouse/all-the-frame-metrics-by-step?fmt=csv'
        resp = get(uri)
        self.history = pd.read_csv(StringIO(resp.text))
        
        self.dqs = pd.read_csv("C:/Users/Justin.Stuck/Desktop/JDQs.csv",low_memory=False)['ticketId']
        
        #transform data from 
        self.history = self.transform(self.history,['latency','bandwidth'])

        self.history = pd.concat([self.history[self.history.name =='Unity Frame Shopping'],self.history[self.history.name =='Unity Frame Tutorial']])
        self.full_history=self.history
        
        
        self.dqs = self.history[self.history['ticketid'].isin(self.dqs)]
        self.dqs = self.dqs[self.dqs.name == 'Unity Frame Shopping']        
        
        
        self.history = self.history[self.history.iscomplete == 1]
        self.respondents = self.history[self.history['studyidguid']==self.studyid.lower()]
        self.shopping = self.respondents[self.respondents.name == 'Unity Frame Shopping']
        self.tutorial = self.respondents[self.respondents.name == 'Unity Frame Tutorial']
        self.tutorial = self.tutorial[self.tutorial.ticketid.isin(self.shopping.ticketid.values)]
        
    def tukey_filter(self, data):
        q1=np.array([self.stats[feat]['Q1'] for feat in self.features])
        q3=np.array([self.stats[feat]['Q3'] for feat in self.features])
        iqr = q3-q1
        lowFences = dict(zip(self.features,q1-1.5*iqr))
        highFences = dict(zip(self.features,q3+1.5*iqr))
        print lowFences
        print highFences
        self.highLat = data[data['latency']>highFences['latency']]   
        self.lowBand = data[data['bandwidth']<lowFences['bandwidth']]
        self.lowFR = data[data['framerate']<self.stats['framerate']['mean']-2*self.stats['framerate']['std']]
        self.highLat['reason'] = 'High Latency'
        self.lowBand['reason'] = 'Low Bandwidth'
        self.lowFR['reason'] = 'Low Frame Rate'
        
        return pd.concat([self.highLat,self.lowBand,self.lowFR])
        
        
        
        
        
        
        
    def logistic_filter(self):
        X, y = self.history[self.features], self.respondents.iscomplete
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=0.05,random_state=0)

        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)
        x = self.respondents['latency']
        y = self.respondents['bandwidth']
        z = self.respondents['framerate']
        
        
        
        
        
        
        
        
    def to_excel(self, removals, filename):
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        
        removals.to_excel(writer, sheet_name='Removal Candidates')
        
        writer.save()






#fc = FrameCleaner('04CE2574-357E-491A-9BD1-51D0A45F32ED')
fc = FrameCleaner('4BC154FD-6429-444E-B589-4B3B7CF1BDA6')
fc.get_data()
#fc.get_removals()

completes = fc.shopping[fc.shopping.iscomplete == 1]
incompletes = fc.shopping[fc.shopping.iscomplete == 0]

#fc.calc_stats(fc.respondents)
data = fc.calc_stats(fc.history)

fc.plot_histos(features=fc.features)
fc.to_excel(fc.tukey_filter(fc.shopping),'frameremovals.xlsx')

print fc.stats























