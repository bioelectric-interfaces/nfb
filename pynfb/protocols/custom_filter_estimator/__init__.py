from os import system

if __name__ == '__main__':
    with open('example.py', 'r', encoding="utf-8") as f:
        get_filter_code = f.read()

    run_get_filter_code = ("\nfrom pynfb.serializers.hdf5 import load_h5py \ndata = 'test'\nfilter = get_filter(data)\nprint('filter saved', filter)")

    with open('_run_get_filter.py', 'w', encoding="utf-8") as f:
        f.write(get_filter_code + run_get_filter_code)

    system("\"E:\_nikolai\programs\WinPython\python-3.4.4.amd64\python\" _run_get_filter.py")