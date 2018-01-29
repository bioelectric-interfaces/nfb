from pynfb.postprocessing.mu_or_not.meta import *
from pynfb.signal_processing.decompositions import ICADecomposition
from mne.viz import plot_topomap
from pynfb.inlets.montage import Montage

group = 'treatment'
subj = 'VV'
day = 1
for day in [1, 2]:
    mat = loadmat(r'C:\Users\Nikolai\Desktop\Liza_diplom_data\Liza_diplom_data\treatment\{0}\bci\{0}_{1}1.mat'.format(subj, day))
    channels = [ch[0] for ch in mat['chan_names'][0]]
    montage = Montage(channels)
    df = pd.DataFrame(data=mat['data_cur'].T, columns=channels)


    df['state'] = mat['states_cur'][0]
    print(df['state'].unique())

    df = df.loc[~get_outliers_mask(df[channels], iter_numb=10, std=3)]

    plt.plot(df[channels])
    plt.show()


    a = QtGui.QApplication([])
    #ica = ICADialog(np.concatenate([df.loc[df['state']==6, channels], df.loc[df['state']==1, channels]]), channels, fs, mode='csp')
    ica = ICADialog(df[channels], channels, fs, mode='ica')
    ica.exec_()
    print(ica.table.get_checked_rows())


    ind = ica.table.get_checked_rows()
    for k in ind:
        plot_topomap(ica.topographies[:, k], montage.get_pos())

    np.save(r'treatment\{0}d{1}_filters.npy'.format(subj, day), ica.unmixing_matrix)
    np.save(r'treatment\{0}d{1}_topographies.npy'.format(subj, day), ica.topographies)
    np.save(r'treatment\{0}d{1}_smr_ind.npy'.format(subj, day), ind)