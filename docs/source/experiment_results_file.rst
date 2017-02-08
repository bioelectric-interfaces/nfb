Results file structure
======================

- **protocol0** (group; initial stats of signals)
- **protocol1** (group; recorded data and signals stats after the first protocol, see :ref:`protocol\<k\><protocolk>`)
- ...
.. _protocolk:
- **protocol\<k\>** (group; recorded data and signals stats after the \<k\>-th protocol)
    * **raw_data** (dataset; raw data recordings except *ignored* channels)
    * **raw_other_data** (dataset; *ignored* channels, for example , for example reference channel)
    * **reward_data** (dataset; reward dinamics time series)
    * **signals_data** (dataset; signals data recordings)
    * **signals_stats** (group; signals stats for every signal)
        - **signal_name1** (group; stats for signal "signal_name1", see **signal_name\<m\>**)
        - **signal_name2** (group; stats for signal "signal_name2", see **signal_name\<m\>**)
        - **...**
        - **signal_name\<m\>** (group; stats for signal "signal_name\<m\>")
            - **mean** (dataset; mean value of signal "signal_name\<m\>")
            - **std** (dataset; std value of signal "signal_name\<m\>")
            - **bandpass** (dataset; bandpass for signal "signal_name\<m\>", for Derived signal only)
            - **spatial_filter** (dataset; spatial filter dataset for signal "signal_name\<m\>", for Derived signal only)
            - **rejections** (group; rejections for signal "signal_name\<m\>", for Derived signal only)
                - **rejection1** (dataset; the first rejection dataset for signal "signal_name\<m\>")
                - **rejection2** (dataset; the second rejection dataset for signal "signal_name\<m\>")
                - **...**
                - **rejection\<j\>** (dataset; the \<j\>-th rejection dataset for signal "signal_name\<m\>")

