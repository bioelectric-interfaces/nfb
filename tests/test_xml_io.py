from pynfb.serializers.xml_ import *

odict = read_xml_to_dict('pynfb/serializers/settings/pilot.xml')
print(odict)
write_dict_to_xml(odict, 'tests/test.xml')