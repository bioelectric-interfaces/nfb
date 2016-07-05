from collections import OrderedDict

general_defaults = OrderedDict([
    #('bDoubleBlind', False),
    #('bShowFBCount', False),
    #('bShowFBRect', False),
    #('fSamplingFrequency', ''),
    #('CompositeMontage', '__'),
    #('CSPSettings', OrderedDict([
    #    ('iNComp', '2'),
    #    ('dInitBand', '8 16')]))
])


vectors_defaults = OrderedDict([
    ('sExperimentName', 'experiment'),
    ('sInletType', 'lsl'),
    ('sStreamName', 'NVX136_Data'),
    ('sRawDataFilePath', ''),
    ('sFTHostnamePort', 'localhost:1972'),
    ('bPlotRaw', 1),
    ('sReference', ''),
    ('vSignals', OrderedDict([
        ('DerivedSignal', [OrderedDict([     # DerivedSignal is list!
            ('sSignalName', 'Signal'),
            ('SpatialFilterMatrix', ''),
            ('bDisableSpectrumEvaluation', 0),
            ('fSmoothingFactor', 0.1),
            ('fFFTWindowSize', 1000),
            ('fBandpassLowHz', 0),
            ('fBandpassHighHz', 250),
            ('fAverage', ''),
            ('fStdDev', ''),
            # ('sType', 'plain')
        ])])])),
    ('vProtocols', OrderedDict([
        ('FeedbackProtocol', [OrderedDict([  # FeedbackProtocol is list!
            ('sProtocolName', 'Protocol'),
            # ('sSignalComposition', 'Simple'),
            # ('nMSecondsPerWindow', ''),
            ('bUpdateStatistics', 0),
            # ('bStopAfter', False),
            # ('bShowFBRect', False),
            ('fDuration', 10),
            # ('fThreshold', ''),
            ('fbSource', 'All'),
            # ('iNComp', ''),
            ('sFb_type', 'Baseline'),
            # ('dBand', ''),
            ('cString', ''),
            ('fBlinkDurationMs', 50),
            ('fBlinkThreshold', 0),
            ('sMockSignalFilePath', ''),
            ('sMockSignalFileDataset', 'protocol1'),
            ('bShowReward', 0)
        ])])])),
    ('vPSequence', OrderedDict([
        ('s', ['Protocol'])])),
])