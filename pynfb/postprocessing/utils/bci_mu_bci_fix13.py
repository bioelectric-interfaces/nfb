import h5py

first_name = r'D:\bci_nfb_bci\bci_nfb_bci\bci_mu_ica_S13_d3_09-27_12-29-02-not-finished_stopped on NFB\experiment_data.h5'
second_name = r'D:\bci_nfb_bci\bci_nfb_bci\bci_mu_ica_S13_d3_continue_09-27_13-34-07\experiment_data.h5'
first = h5py.File(first_name)
second = h5py.File(second_name)
fixed = h5py.File('bci_mu_ica_S13_d3_fixed3.h5', 'a')

n_protocols = 37
n_recorded = [first['protocol{}'.format(k+1)].attrs['name'] for k in range(len(first)-3)].index('Bci') + 1
print(n_recorded)


for j in range(1, n_protocols + 1):
    if j <= n_recorded:
        fixed['protocol{}'.format(j)] = h5py.ExternalLink(first_name, 'protocol{}'.format(j))
    else:
        fixed['protocol{}'.format(j)] = h5py.ExternalLink(second_name, 'protocol{}'.format(j-1))



first.close()
second.close()
fixed.close()