# Big Sur OpenGL bug workaround (https://bugs.python.org/issue41100) ---------------------------------------------------
try:
    import OpenGL as ogl
    try:
        import OpenGL.GL   # this fails in <=2020 versions of Python on OS X 11.x
    except ImportError:
        from ctypes import util
        orig_util_find_library = util.find_library
        def new_util_find_library( name ):
            res = orig_util_find_library( name )
            if res: return res
            return '/System/Library/Frameworks/'+name+'.framework/'+name
        util.find_library = new_util_find_library
except ImportError:
    pass
# ----------------------------------------------------------------------------------------------------------------------


import os
from importlib import import_module
STATIC_PATH = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/static')

__version__ = "0.1.0"

# pylsl has a dll order problem with kiwisolver (which is used by matplotlib). If kiwisolver is first imported before
# pylsl, pylsl will not find any streams. Import this package (pynfb) in your entry point to work around the problem.
import_module("pylsl")
import_module("kiwisolver")
