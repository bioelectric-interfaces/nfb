import expyriment
import expyriment._internals as _internals


def strip_and_escape(func):
    def stripped_and_escaped(*args, **kwargs):
        out = func(*args, **kwargs)
        strip_and_escape_string = lambda s: s.rstrip('\n').encode('unicode-escape')
        return map(strip_and_escape_string, out)
    return stripped_and_escaped


_internals.import_plugin_defaults = strip_and_escape(_internals.import_plugin_defaults)
_internals.import_plugin_defaults_from_home = strip_and_escape(_internals.import_plugin_defaults_from_home)
