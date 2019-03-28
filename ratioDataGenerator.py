import pandas as pd
import numpy as np
import os
import scikits.bootstrap as boot
from scipy.stats import kurtosis
from scipy.stats import skew

projectPath = 'F:\\Dropbox\\SSE\\PROJECTS' 

df = pd.read_pickle('1926_FF_daily_merged_factors.pkl')
cols = df.columns[1:] # Only select the numerical columns
df[cols] = np.log(1 + df[cols] ) # Convert to log(R)

"""
Select input and name of output
['Mkt-RF', 'SMB', 'HML', 'Mom']
"""
cols = 'Mom' # Select column
nam = "MOM"  # Output name

"""
Functions
"""
def momentRatio(X):
    """
    Following Lo and Mackinley (1988)
    NOTE: Need to define the period length, p, outside the function
    """  
    th_i = np.mean(X[:, 0])*p
    th_h = np.mean(X[:, 1])
    ER = th_h - th_i
    
    th_i = np.var(X[:, 0], ddof=1)*p
    th_h = np.var(X[:, 1], ddof=1)
    VR = th_h - th_i
    
    th_i = skew(X[:, 0], bias=False)*(1/np.sqrt(p))
    th_h = skew(X[:, 1], bias=False)
    SR = th_h - th_i
    
    th_i = kurtosis(X[:, 0], bias=False)*(1/p)
    th_h = kurtosis(X[:, 1], bias=False)
    KR = th_h - th_i
    
    return([ER, VR, SR, KR])

"""
Setup range of comparisons, and correct names of each range
"""
freq = ['annualy', 'biannual', 'quarterly', 'monthly', 'weekly', 'daily']
H_sel = [range(264, 10*264+1, 264), range(132, 10*264+1, 132), 
         range(66, 2*264+1, 66), range(22, 2*264+1, 22), range(5, 2*264+1, 5), 
         range(1, 265, 1) ]

a = [0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995]
B = 10000
keys = ['q0.5', 'q2.5', 'q5', 'q50', 'q95', 'q97.5', 'q99.5']

for i, H in enumerate(H_sel):
    print(nam+'_'+freq[i]+' '+str(H))
    
    X = pd.DataFrame(df[cols].rolling(H[0]).sum(), copy = True)
    X.columns = [nam+str(H[0])]
    
    out_er = pd.DataFrame([], index = keys)
    out_vr = pd.DataFrame([], index = keys)
    out_sr = pd.DataFrame([], index = keys)
    out_kr = pd.DataFrame([], index = keys)
    
    for j, h in enumerate(H[1:]):
        print('j: ', j, ' h: ', h)
        X[nam+str(h)] = df[cols].rolling(h).sum()
        
        p = j + 2
        boot_est = boot.ci(data = X.loc[:, [nam+str(H[0]), nam+str(h)]].dropna(), statfunction = momentRatio, alpha = a, n_samples = B, method = 'bca')
        out_er['h'+str(p)] = boot_est[:, 0] # First column are the ER CI values
        out_vr['h'+str(p)] = boot_est[:, 1]
        out_sr['h'+str(p)] = boot_est[:, 2]
        out_kr['h'+str(p)] = boot_est[:, 3]
        
    filename = '. - Term Structure\\Code\\Output\\Ratio Test Estimates\\' + nam + '_' + freq[i] + '_er.csv'
    out_er.T.to_csv(os.path.join(projectPath, filename), index = True)
    
    filename = '. - Term Structure\\Code\\Output\\Ratio Test Estimates\\' + nam + '_' + freq[i] + '_vr.csv'
    out_vr.T.to_csv(os.path.join(projectPath, filename), index = True)
    
    filename = '. - Term Structure\\Code\\Output\\Ratio Test Estimates\\' + nam + '_' + freq[i] + '_sr.csv'
    out_sr.T.to_csv(os.path.join(projectPath, filename), index = True)
    
    filename = '. - Term Structure\\Code\\Output\\Ratio Test Estimates\\' + nam + '_' + freq[i] + '_kr.csv'
    out_kr.T.to_csv(os.path.join(projectPath, filename), index = True)

