from pynfb.io.xml_ import *

odict = read_xml_to_dict('pynfb/io/settings/pilot.xml')
print(odict)
write_dict_to_xml(odict, 'tests/test.xml')