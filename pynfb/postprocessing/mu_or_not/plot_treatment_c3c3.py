import pandas as pd
import seaborn as sns
import pylab as plt

df = pd.read_pickle('treatment_c3c4.pkl')
print(df)

#df['session'] = (df['session']-1)//3+1
for subj in df['subj'].unique():
    df.loc[(df['subj'] == subj) & (df['type'] == 'c3env') & (df['session']<300), 'val'] /= df.loc[
        (df['subj'] == subj) & (df['type'] == 'c3env') & (df['session']<4), 'val'].mean()
    df.loc[(df['subj'] == subj) & (df['type'] == 'c4env') & (df['session']<300), 'val'] /= df.loc[
        (df['subj'] == subj) & (df['type'] == 'c4env') & (df['session']<4), 'val'].mean()
#    df.loc[(df['subj'] == subj) & (df['type'] == 'c3env') & (df['session'] > 5), 'val'] /= df.loc[
#        (df['subj'] == subj) & (df['type'] == 'c3env') & (df['session'] == 6), 'val'].mean()
#    df.loc[(df['subj'] == subj) & (df['type'] == 'c4env') & (df['session'] > 5), 'val'] /= df.loc[
#        (df['subj'] == subj) & (df['type'] == 'c4env') & (df['session'] == 6), 'val'].mean()

df['session'] = (df['session']-1)//3+1
sns.pointplot('session', 'val', 'type', df, ci=75, dodge=True)
sns.swarmplot('session', 'val', 'type', df, alpha=0.3)
plt.ylim(0.25, 1.75)
plt.title('Control')
plt.show()

from scipy.stats import ttest_1samp, ttest_ind

plt.plot([ttest_ind(df.loc[(df['session']==k+1) & (df['type']=='c4env'), 'val'], df.loc[(df['session']==6) & (df['type']=='c4env'), 'val']).pvalue for k in range(10)])
plt.plot([ttest_ind(df.loc[(df['session']==k+1) & (df['type']=='c3env'), 'val'], df.loc[(df['session']==6) & (df['type']=='c3env'), 'val']).pvalue for k in range(10)])
plt.show()
print(ttest_1samp(df.loc[df['session']==1, 'val'], 1))