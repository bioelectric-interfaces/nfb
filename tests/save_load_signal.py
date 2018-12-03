from pynfb.serializers.xml_ import save_signal, load_signal

from pynfb.signals import DerivedSignal

#save_signal(DerivedSignal(), 'sig.xml')
load_signal('sig.xml', ['ch'+str(j) for j in range(50)])