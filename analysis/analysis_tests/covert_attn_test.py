from utils.load_results import load_data
import pandas as pd
import plotly_express as px
import analysis.analysis_functions as af

task_data = {}
h5file = "/Users/christopherturner/Documents/EEG_Data/system_testing/attn_right_02-01_15-44-49/experiment_data.h5"

# Put data in pandas data frame
df1, fs, channels, p_names = load_data(h5file)
df1['sample'] = df1.index

channels.append("signal_AAI_sc")
channels.append("signal_AAI_so")

df2 = pd.melt(df1, id_vars=['block_name', 'block_number', 'sample', 'choice', 'probe', 'answer', 'reward'],
                  value_vars=channels, var_name="channel", value_name='data')
aai_sc_data = df2.loc[df2['channel'] == "signal_AAI_sc"].reset_index(drop=True)
aai_so_data = df2.loc[df2['channel'] == "signal_AAI_so"].reset_index(drop=True)

fig = px.line(aai_sc_data, x=aai_sc_data.index, y="data", color='block_name', title=f"scalp aai")
fig.show()
fig = px.line(aai_so_data, x=aai_so_data.index, y="data", color='block_name', title=f"scalp aai")
fig.show()
fig = px.box(aai_sc_data, x='block_name', y="data", title="scalp aai")
fig.show()
fig = px.box(aai_so_data, x='block_name', y="data", title='source aai')
fig.show()
pass