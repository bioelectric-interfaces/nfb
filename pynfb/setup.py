from cx_Freeze import setup, Executable

setup(
    name = "pynfb",
    version = "0.1",
    description = "Python NFB",
    executables = [Executable("main.py")]
)