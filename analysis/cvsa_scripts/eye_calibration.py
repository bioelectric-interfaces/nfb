"""
Script to get the screen witdh from the eye calibration
"""
from utils.load_results import load_data
from scipy.signal import butter, lfilter, freqz

import plotly.graph_objs as go

# ------ low pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def eye_calibration(h5file, plot=False):
    df1, fs, channels, p_names = load_data(h5file)
    df1['sample'] = df1.index

    cutoff = 10
    df1['ECG_FILTERED'] = butter_lowpass_filter(df1['ECG'], cutoff, fs)
    df1['EOG_FILTERED'] = butter_lowpass_filter(df1['EOG'], cutoff, fs)

    # Get the left mid side (idx=13) and right mid side (idx=15)
    eye_calib_data = df1[df1['block_name'] == 'EyeCalib']
    left_calib_data = eye_calib_data[eye_calib_data['probe'] == 13]
    centre_calib_data = eye_calib_data[eye_calib_data['probe'] == 14]
    right_calib_data = eye_calib_data[eye_calib_data['probe'] == 15]

    buff = 750 # Remove the eye movement stage
    left_calib_mean = (left_calib_data['EOG_FILTERED'] - left_calib_data['ECG_FILTERED'])[buff:].mean()
    centre_calib_mean = (centre_calib_data['EOG_FILTERED'] - centre_calib_data['ECG_FILTERED'])[buff:].mean()
    right_calib_mean = (right_calib_data['EOG_FILTERED'] - right_calib_data['ECG_FILTERED'])[buff:].mean()

    eye_centre = centre_calib_mean
    eye_range = left_calib_mean - right_calib_mean

    print(f"EYE CENTRE: {eye_centre}, EYE RANGE: {eye_range}")

    if plot:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df1.index, y=df1['ECG_FILTERED'],
                            mode='lines',
                            name='ECG'))
        fig1.add_trace(go.Scatter(x=df1.index, y=df1['EOG_FILTERED'],
                            mode='lines',
                            name='EOG'))
        fig1.add_vrect(x0=left_calib_data['sample'].iloc[0], x1=left_calib_data['sample'].iloc[-1],
                      annotation_text="left", annotation_position="top left",
                      fillcolor="green", opacity=0.25, line_width=0)
        fig1.add_vrect(x0=right_calib_data['sample'].iloc[0], x1=right_calib_data['sample'].iloc[-1],
                      annotation_text="right", annotation_position="top left",
                      fillcolor="red", opacity=0.25, line_width=0)

        fig1.show()


        left_eye_sig = (left_calib_data['EOG_FILTERED'] - left_calib_data['ECG_FILTERED'])[750:]
        # left_eye_sig = left_eye_sig.reset_index()
        right_eye_sig = (right_calib_data['EOG_FILTERED'] - right_calib_data['ECG_FILTERED'])[750:]
        # right_eye_sig = right_eye_sig.reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=left_eye_sig.index, y=left_eye_sig,
                            mode='lines',
                            name='left'))
        fig2.add_trace(go.Scatter(x=right_eye_sig.index, y=right_eye_sig,
                            mode='lines',
                            name='right'))
        fig2.show()

    return eye_centre, eye_range

if __name__ == "__main__":
    # -------------
    h5file = "/Users/christopherturner/Documents/EEG_Data/cvsa_pilot_testing/lab_test_20220428/0-eye_calibration_ct_test_04-28_17-20-40/experiment_data.h5"  # Horizontal 9 pt calibration
    # h5file = "/Users/2354158T/OneDrive - University of Glasgow/Documents/cvsa_pilot_testing/lab_test_20220428/0-eye_calibration_ct_test_04-28_17-20-40/experiment_data.h5"
    eye_centre, eye_range = eye_calibration(h5file)