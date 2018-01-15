import seaborn as sns
import pandas as pd
import numpy as np
import json
import pylab as plt
from scipy.stats import linregress

for k in range(60, 70):
    df = pd.read_csv('stats_slopes{}.csv'.format(k))
    stds = pd.DataFrame(columns=['std', 'group'])
    for g, group in enumerate(['Real', 'Mock']):
        for s, subj in enumerate(df.loc[df.group == group, 'subj'].unique()):

            for d, day in enumerate(df.loc[(df.group == group) & (df.subj == subj), 'day'].unique()):
                stds.loc[len(stds)] = {'std': df.loc[(df.group == group) & (df.subj == subj) & (df.day == day), 'slope'].std(), 'group': group}

    sns.barplot(x='group', y='std', data=stds, estimator=np.median)
    sns.swarmplot(x='group', y='std', data=stds, color='r')
    plt.show()

    from scipy.stats import *
    #print(bartlett(df.loc[(df.group == 'Real'), 'slope'], df.loc[(df.group == 'Mock'), 'slope']))
    #print(bartlett(df.loc[(df.group == 'Real'), 'slope'], df.loc[(df.group == 'Real'), 'slope']))
    print(k, levene(df.loc[(df.group == 'Real'), 'slope'], df.loc[(df.group == 'Mock'), 'slope']))
    #print(levene(df.loc[(df.group == 'Real'), 'slope'], df.loc[(df.group == 'Real'), 'slope']))
    #print(normaltest(df.loc[(df.group == 'Real'), 'slope']))
    #print(normaltest(df.loc[(df.group == 'Mock'), 'slope']))