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
    ('vSignals', OrderedDict([
        ('DerivedSignal', [OrderedDict([     # DerivedSignal is list!
            ('sSignalName', 'Signal'),
            ('SpatialFilterMatrix', ''),
            ('fBandpassLowHz', ''),
            ('fBandpassHighHz', ''),
            ('fAverage', ''),
            ('fStdDev', ''),
            # ('sType', 'plain')
        ])])])),
    ('vProtocols', OrderedDict([
        ('FeedbackProtocol', [OrderedDict([  # FeedbackProtocol is list!
            ('sProtocolName', 'Protocol'),
            # ('sSignalComposition', 'Simple'),
            # ('nMSecondsPerWindow', ''),
            ('bUpdateStatistics', False),
            # ('bStopAfter', False),
            # ('bShowFBRect', False),
            ('fDuration', '10'),
            # ('fThreshold', ''),
            ('fbSource', ''),
            # ('iNComp', ''),
            ('sFb_type', ''),
            # ('dBand', ''),
            ('cString', '')
        ])])])),
    ('vPSequence', OrderedDict([
        ('s', ['Protocol1'])])),
])