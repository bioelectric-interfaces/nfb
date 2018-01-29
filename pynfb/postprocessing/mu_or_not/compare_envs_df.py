import pandas as pd
import pylab as plt
import numpy as np
import seaborn as sns
subjects = ['p6', 'p8', 'p10', 'p13', 'VV', 'p4', 'IO', 'KM']


df = pd.DataFrame(columns=['val', 'type', 'subj', 'day', 'session'])
for subj in subjects:
    for day in [1, 2]:
        dd = pd.read_pickle('{}day{}.pkl'.format(subj, day))[['SMRRightEnv', 'SMRLeftEnv', 'C3env', 'C4env', 'session']]

        dd = dd.loc[dd['session']>0]
        #dd['session'] = dd['session'] + 15
        #dd = dd.append(dd2)

        dd = dd.groupby('session').mean()
        dd['session'] = (dd.index - 1) // 3 + 1
        dd.loc[:, ['SMRRightEnv', 'SMRLeftEnv', 'C3env', 'C4env']] /= dd.loc[
            dd['session'] == 1, ['SMRRightEnv', 'SMRLeftEnv', 'C3env', 'C4env']].mean()
        print(dd)
        for val in ['SMRRightEnv', 'SMRLeftEnv', 'C3env', 'C4env']:
            df = df.append(pd.DataFrame({'val': dd[val], 'subj': subj, 'day': day, 'session': dd['session'] + (day-1)*5, 'type':val}))

#for subj in subjects:
#    df1 = df.loc[df['subj']==subj]
#    sns.boxplot('session', 'val', 'type', data=df1)
#    plt.show()

sns.boxplot('session', 'val', 'type', data=df)
plt.show()