from pynfb.signals import DerivedSignal, CompositeSignal
from pynfb.serializers.hdf5 import save_signals

signals = [DerivedSignal(name='One'),
           DerivedSignal(name='Two')]
signals += [CompositeSignal(signals, 'One+Two', name='Three')]

save_signals('signals_info.hdf5', signals, group_name='protocol1')