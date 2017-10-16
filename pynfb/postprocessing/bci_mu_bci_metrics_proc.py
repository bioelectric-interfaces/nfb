import pandas as pd
from scipy.stats import linregress
import pylab as plt
import seaborn as sns
import numpy as np

from scipy import stats


def get_acc_slopes(filename, n_subj):
    df = pd.read_csv(filename)
    slopes = []
    acc_diff = []
    for day in range(3):
        for subj in range(n_subj):

            if any(df[df['subj'] == subj]['day'] == day):
                avg_before = df[df['subj'] == subj][df['day'] == day][df['after'] == 0].mean()
                avg_after  = df[df['subj'] == subj][df['day'] == day][df['after'] == 1].mean()
                slopes.append(avg_before['fb_slope'] if avg_before['fb_slope']>-0.1 else 0)
                acc_diff.append(avg_after['acc_train'] - avg_before['acc_train'])
    return np.array(slopes), np.array(acc_diff)

slopes, acc_diff = get_acc_slopes('bcinfbbci_metrics.csv', 7)
slopes_mock, acc_diff_mock = get_acc_slopes('bcinfbbci_mock_metrics.csv', 7)

f, axes = plt.subplots(1, 3)

sns.regplot(slopes, acc_diff, ax=axes[0])
sns.regplot(slopes_mock, acc_diff_mock, ax=axes[0])
axes[0].set_title('p_real={:.3f}\np_mock={:.3f}'.format(stats.linregress(slopes, acc_diff).pvalue, stats.linregress(slopes_mock, acc_diff_mock).pvalue))
axes[0].legend(['real', 'mock'])
axes[0].set_xlim(-0.003, 0.003)
axes[0].set_ylim(-0.15, 0.15)

sns.boxplot(x=np.concatenate([['real']*len(slopes), ['mock']*len(slopes_mock)]), y=np.concatenate([slopes, slopes_mock]), ax=axes[1])
axes[1].set_ylabel('slope')
axes[1].set_title('p={:.3f}'.format(stats.ttest_ind(slopes, slopes_mock)[1]))
axes[1].legend(['real', 'mock'])

sns.boxplot(x=np.concatenate([['real']*len(acc_diff), ['mock']*len(acc_diff_mock)]), y=np.concatenate([acc_diff, acc_diff_mock]), ax=axes[2])
axes[2].set_ylabel('acc_diff')
axes[2].set_title('p={:.3f}'.format(stats.ttest_ind(acc_diff, acc_diff_mock)[1]))
axes[2].legend(['real', 'mock'])

plt.tight_layout()
plt.show()

