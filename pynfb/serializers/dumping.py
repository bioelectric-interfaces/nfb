import pickle


def dump_object(obj, file_name):
    """ Dump simple object (e.g. BCISignal)
    :param obj: object to dump
    :param file_name: file path name
    :return: None
    """
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_object(file_name):
    """ Load simple object (e.g. BCISignal)
    :param file_name: file path name
    :return: object loaded
    """
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj


if __name__ == '__main__':
    obj = {'a': 5, 'b': [1, 2]}
    file_name = 'test_dump.pkl'
    dump_object(obj, file_name)
    obj2 = load_object(file_name)
    print(obj, obj2)