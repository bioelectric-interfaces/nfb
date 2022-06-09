;aim: wait for an incoming signal from E-prime/StimPC via DigInput. TBD.
      ;conditional on that input ramp up either a low freq or high freq waveform, 
;hold for xy s, set amplitude to zero and go back to waiting. Check.



;***** set up variables and 1401 **********************
        SET     10.000 1 0     ;10 milliseconds per step
	DIGOUT 	[00000000]              ;written as marker on Spike
	DIGLOW 	[00000000]              ;for EEG		

;***** experiment ************************************

IDLE:   DIBEQ   [......00], IDLE       ;see manual example
        DISBEQ   [......01], PULSE
        JUMP   IDLE


PULSE:  'G  DIGOUT [.......1]      ;Pulse outputs      >S to stop
            DIGOUT [.......0]      ;Set output low     >S to stop 
        MARK    49                      ;prints "1"
        JUMP    IDLE         


AKEY:	'a      JUMP PULSE 	        ;key press initiates trial

	