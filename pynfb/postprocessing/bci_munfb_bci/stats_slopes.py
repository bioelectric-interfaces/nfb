import seaborn as sns
import pandas as pd
import numpy as np
import json
import pylab as plt
from scipy.stats import linregress

work_dir = r'C:\Users\Nikolai\Desktop\bci_nfb_bci\bci_nfb_bci'

def open_desc(group='Real'):
    desc_file = 'info_mock.json' if group == 'Mock' else 'info.json'
    with open('{}/{}'.format(work_dir, desc_file), encoding="utf-8") as f:
        desc = json.loads(f.read())
    return desc

fs = 500
cm = sns.color_palette()
n_slopes = 100

stats = pd.DataFrame(columns=['group', 'subj', 'day', 'slope', 'p-value'])
for k in range(30, 60, 10):
    print(k)
    for group in ['Real', 'Mock']:
        desc = open_desc(group)
        print(desc)

        n_samples = fs * k
        for subj, days in enumerate(desc['subjects'][:]):
            for day, exp_name in enumerate(days[:]):
                print(group, subj, day)
                exp_data_path = '{}\{}.pkl'.format(work_dir, exp_name)
                df = pd.read_pickle(exp_data_path, 'gzip')
                #df['before'] = df['block_number'] <= 12
                #df['logenv'] = np.log(df['env'])
                norm = df.loc[df.block_name == 'Baseline', 'env'].median()
                fb = df.loc[df.block_name == 'FB', 'env'].as_matrix() / norm
                #plt.plot(fb)
                #plt.title(exp_name)
                #plt.show()
                for s in range(n_slopes):
                    start = np.random.randint(0, len(fb) - n_samples)
                    ind = np.arange(start, start + n_samples)
                    slope = linregress(ind / fs, fb[ind]).slope
                    p_val = linregress(ind / fs, fb[ind]).pvalue
                    stats.loc[len(stats)] = {'group': group, 'subj': subj+1, 'day': day+1, 'slope': slope, 'p-value':p_val}
                    #print(slope, p_val)
                    #sns.regplot(ind / fs, fb[ind])
                    #plt.show()



    stats.to_csv('stats_slopes{}.csv'.format(k), index=False)