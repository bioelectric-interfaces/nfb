import pandas as pd
from scipy.stats import linregress
import pylab as plt
import seaborn as sns

df = pd.read_csv('bcinfbbci_metrics.csv')

subj = 5
day = 2

avg_before = df[df['subj'] == subj][df['day'] == day][df['after'] == 0].mean()
avg_after  = df[df['subj'] == subj][df['day'] == day][df['after'] == 0].mean()
slope = avg_before['fb_slope']
acc_train_diff = avg_before[['acc_train', 'acc_train0', 'acc_train1', 'acc_train2']]
acc_test_diff = avg_before[['acc_test', 'acc_test0', 'acc_test1', 'acc_test2']]
print(acc_train_diff)
print(acc_test_diff)