import numpy as np
import pandas as pd
import plotly_express as px


# Get x values of the sine wave

time = np.arange(0, 1000, 0.5)

# Amplitude of the sine wave is sine of a variable like time

n1 = np.random.normal(scale=0.75, size=time.size)
freq = 10
alpha1 = np.sin(time*(2*np.pi)/freq) + 11+n1

n2 = np.random.normal(scale=0.75, size=time.size)
alpha2 = np.sin(time*(2*np.pi)/freq) + 10+n2

aai = (alpha1-alpha2)/(alpha1+alpha2)

df = pd.DataFrame([alpha1, alpha2, aai]).T
df.columns = ['alpha1', 'alpha2', 'aai']

# Plot a sine wave using time and amplitude obtained for the sine wave
fig = px.line(df, title='x')
fig.show()

