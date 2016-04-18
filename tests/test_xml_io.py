from pynfb.experiment_parameters.xml_io import *

odict = read_xml_to_dict('pynfb/experiment_parameters/settings/pilot.xml')
print(odict)
write_dict_to_xml(odict, 'tests/test.xml')