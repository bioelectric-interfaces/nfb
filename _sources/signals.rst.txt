Derived signal settings
=======================

.. image:: signal.png
   :width: 300


Signal Settings include:

**Name**: name of the signal.

**Spatial filter**: spatial filter can be chosen through the list of available filters or entered manually ("CUSTOM" type). A current signal sample is computed as a weighted sum of the raw samples, and that weights are construct the spatial filter. In the case of the custom filter, enter it as following: Cz=1, F4=1 (others will not be processed).

**Temporal settings**: settings of the temporal filter.

   **Type**: type of the filter. envdetector - signal will be filtered and the envelope will be computed. filter - only temporal filtering without the envelope. identity - raw siganl. Accordingly to the type, the following settings will be available or not.
   
   **Band**: frequency filter border.
   
   **Filter type**: FFT, butter (for Butterworth), complexdem, cfir.
   
   **Window size [samp.]**: the size of the window in samples used for FFT and CFIR.
   
   **Filter order**: order of the filter used for Butterworth and complexdem.
   
   **Smoother type**: exponential ("exp") or Savitzky–Golay (savgol) smoother.
   
   **Smoother factor**: factor used in the exponential smoothing. Higher values correspond to a higher smoothness of the signal.
   
   **Artif. delay [ms]**: artificial delay, used to add a latency between the signal and the feedback presentation.

**BCI mode**: the mode for using imaginary movements (in testing).

**Save**: saving of the signal.
