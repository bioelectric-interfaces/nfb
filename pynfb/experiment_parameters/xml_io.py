import xmltodict


def read_xml_to_dict(filename, skip_root=False):
    """ Read xml to ordered dict
    :param filename: path to file
    :param skip_root: if True skip root
    :return: OrderedDict instance
    """
    # postprocessor convert to int if possible
    def postprocessor(path, key, value):
        try:
            return key, int(value)
        except (ValueError, TypeError):
            if value is None:
                value = ''
            return key, value
    # read and parse
    with open(filename, 'r') as f:
        d = xmltodict.parse(f.read(), postprocessor=postprocessor)
    if skip_root:
        d = list(d.values())[0]
    return d


def write_dict_to_xml(dict_, filename):
    """ Write dict to xml
    :param dict_: OrderedDict instance
    :param filename: path to file
    :return: None
    """
    # unparse and write
    with open(filename, 'w') as f:
        f.write(xmltodict.unparse(dict_, pretty=True))
    pass


if __name__ == '__main__':
    odict = read_xml_to_dict('settings/pilot.xml')
    print(odict)
    write_dict_to_xml(odict, 'settings/test.xml')