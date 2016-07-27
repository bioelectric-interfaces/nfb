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


def save_spatial_filter(file_path, filter_, channels_labels=None):
    """
    Save spatial filter to file
    :param file_path: path to file
    :param filter_: filter coefficient array or list
    :param channels_labels: labels for channels; line format is if it's not None  '<label> <value>', else just '<value>'
    :return:
    """
    filter_str = [val + '\n' for val in array(filter_).astype(str)]
    with open(file_path, 'w') as f:
        if channels_labels is None:
            f.writelines(filter_str)
        else:
            f.writelines([label+' '+val_str for label, val_str in zip(channels_labels, filter_str)])


if __name__ == '__main__':
    labels = ['Cz', 'Pz', 'Fp1']
    save_spatial_filter('spatial_test.txt', [1, 2, 3], labels[-1::-1])
    print(read_spatial_filter('spatial_test.txt', labels))