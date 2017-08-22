from cx_Freeze import setup, Executable
from cx_Freeze.finder import BUILD_LIST

setup(
    name = "nfb_lab_bci_sim",
    version = "0.1",
    description = "NFB Lab BCI lsl stream simulation",
    options = {"build_exe": {"packages": ["numpy"]}},
    executables = [Executable("bci_lsl_stream.py")]
)