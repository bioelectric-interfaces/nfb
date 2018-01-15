import seaborn as sns
import pandas as pd
import numpy as np
import json
import pylab as plt
from scipy.stats import linregress

df = pd.read_csv('stats_slopes50.csv')
#df = df[df.group == 'Mock']
plt.hist(df.loc[(df.group == 'Real') & (df['p-value']<2), 'slope'], bins=np.linspace(-1, 1, 100), density=True, alpha=0.5)
#plt.hist(df.loc[(df.group == 'Mock') & (df['p-value']<2), 'slope'], bins=np.linspace(-1, 1, 100), density=True, alpha=0.5)
#sns.kdeplot(df.loc[(df.group == 'Mock') & (df['p-value']<2), 'slope'])
plt.show()


fig, axes = plt.subplots(2, 10)
stds = pd.DataFrame(columns=['std', 'group'])
for g, group in enumerate(['Real', 'Mock']):
    for s, subj in enumerate(df.loc[df.group == group, 'subj'].unique()):
        axes[0, s].set_title('S'+str(s))
        for d, day in enumerate(df.loc[(df.group == group) & (df.subj == subj), 'day'].unique()):
            stds.loc[len(stds)] = {'std': df.loc[(df.group == group) & (df.subj == subj) & (df.day == day), 'slope'].std(), 'group': group}
        axes[g, s].hist(df.loc[(df.group == group) & (df.subj == subj), 'slope'], np.linspace(-0.7, 0.7, 50), density=True)

axes[0, 0].set_ylabel('Real')
axes[1, 0].set_ylabel('Mock')

#sns.pairplot(df, 'subj', vars=['slope'])
plt.show()

sns.barplot(x='group', y='std', data=stds, estimator=np.median)
sns.swarmplot(x='group', y='std', data=stds, color='r')
plt.show()

sns.kdeplot(df.loc[(df.group == 'Real'), 'slope'])
sns.kdeplot(df.loc[(df.group == 'Mock'), 'slope'])
plt.show()

from scipy.stats import *
print(bartlett(df.loc[(df.group == 'Real'), 'slope'], df.loc[(df.group == 'Mock'), 'slope']))
print(bartlett(df.loc[(df.group == 'Real'), 'slope'], df.loc[(df.group == 'Real'), 'slope']))
print(levene(df.loc[(df.group == 'Real'), 'slope'], df.loc[(df.group == 'Mock'), 'slope']))
print(levene(df.loc[(df.group == 'Real'), 'slope'], df.loc[(df.group == 'Real'), 'slope']))
print(normaltest(df.loc[(df.group == 'Real'), 'slope']))
print(normaltest(df.loc[(df.group == 'Mock'), 'slope']))