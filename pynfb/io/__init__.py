from numpy import array


def read_spatial_filter(filepath, channel_labels=None):
    """
    Read spatial filter from path to file
    :param channel_labels: channel labels list
    :param filepath: path to file
    :return: spatial filter
    """
    with open(filepath, 'r') as f:
        lines = array([l.split() for l in f.read().splitlines()])
    if len(lines[0]) == 1:
        _filter = lines.astype(float).flatten()
    elif len(lines[0]) == 2:
        if channel_labels is None:
            raise ValueError ('Channels labels is None but spatial filter file contains labels')
        _filter_dict = dict(zip([l.upper() for l in lines[:, 0]], lines[:, 1].astype(float)))
        _filter = array([_filter_dict.get(label.upper(), 0) for label in channel_labels])
    else:
        raise ValueError ('Empty file or wrong format')
    return _filter

if __name__ == '__main__':
    read_spatial_filter('settings\\n_s')