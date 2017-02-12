Experimental designer
=====================

.. image:: design.png
   :width: 600
This module allows you to configure the experiment design. Customized design is saved in an .xml file and can be
loaded with further use of the program. ‘Start’ button starts the module of carrying out experiments in which an
experiment is conducted with your settings.

Below is a simplified version that describes the basic settings of the experiment in order from top to bottom and left to right.

**Name**: the name of the experiment (in a folder ‘results’ there is a folder with the same name and added timestamp).

**Inlet**: selection of the data stream to which you want to connect. There is a choice of four options: Normal LSL stream (for connection of devices with LSL support), LSL generator (created LSL flow with a model signal for the test program), LSL of file (created LSL signal playback stream recorded in the file during the previous experiments), FieldTripBuffer (connection for FieldTripBuffer Protocol).

**Reference**: a list of channels that should not be taken into account when conducting the experiment (in the construction of spatial filters).

**Plot raw**: disables / enables online drawing of raw signals.

**Plot signals**: disables / enables online drawing of processed signals.

**Reward period**: the period of accrual encouragement (if the processed signal from the test exceeds a predetermined threshold, then the subject is beginning to accrued encouragement points with the given period).

**Test beep sound**: check of the sound stimulus.

**Signals list**: configuration of the processed signals (such signals obtained from raw signals through the application of spatial filters, frequency filters, amplitudes calculation, subtracting the mean and dividing by the standard deviation, thus the smoothing is performed). When you double-click on one of the signals you open the composite signal settings:

.. toctree::
   signals

**Composite signals**: configurations of composite signals (samples of these signals are obtained by means of an algebraic expression of the signals samples from the field of “Signals”). When you double-click on one of the signals you open the settings of the composite signal:

.. toctree::
   composite

**Protocols list**: includes settings of protocols (experiment consists of several protocols, each of which has a duration, settings of signal processing and rendering properties window for subject). When you double-click on one of the protocols you open protocol settings.

.. toctree::
   protocol

**Protocol sequence**: here you set the protocol sequence that will be implemented during the experiment (minutes can be dragged from the field “Protocols”).

The **Start** button starts the experiment.