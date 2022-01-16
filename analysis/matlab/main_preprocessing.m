
%% 1. SETTINGS

% 1. Set current directory

clear all

%Directory with data, name of participant, name of recording
data_directory =                '/Users/christopherturner/Documents/GitHub/nfb/analysis'
currentParticipant =            'ct_102';
recordingEnding =               '_1';

dataSet =                       strcat('sh_scalp_0-nfb_task_SH01_01-11_15-50-56-bv','.vhdr');
dataSetFull =                   strcat(data_directory, '/');
%currentPath =                   strcat('C:\Users\simonha\OneDrive - University of Glasgow\research\data','\',currentParticipant);
%chanLocPath =                   'C:\Users\simonha\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\EEGLAB\plugins\dipfit4.3\standard_BEM\elec\standard_1005.elc';
%fieldTripPath =                 'C:\Users\simonha\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\FieldTrip';
%eegLabPath =                    'C:\Users\simonha\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\EEGLAB';
addpath(data_directory)%, eegLabPath, fieldTripPath)
cd(dataSetFull);

% 2. Open eeglab
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
close all
% 3. Load participant

%Read in file
EEG = pop_loadbv(dataSetFull, dataSet);
%Check consistency
EEG = eeg_checkset(EEG);
%Read in channel locs
EEG.chalocs = readlocs(chanLocPath);


%% 2. Preprocessing


%%%%%%%%%%%%%%
% 1. Detrend
%%%%%%%%%%%%%%
EEG.data = detrend(EEG.data);

%%%%%%%%%%%%%%
% 2. Filter
%%%%%%%%%%%%%%
%alternative: (EEG, X, X, [], 1, [], 0) for notch filter

EEG = pop_eegfiltnew(EEG, 0.1, 40, [], 0, [], 0); %bandpass 0.1 - 40Hz 

%%%%%%%%%%%%%%
% 3. Epoch
%%%%%%%%%%%%%%
EEG = pop_epoch(EEG, {'S  1'  'S  2'  'S  3' 'S  4' 'S  5'  'S  6' ...
    'S  7' 'S  8' 'S  9' 'S 10'}, [-2  2], 'epochinfo', 'yes');

%%%%%%%%%%%%%%
% 4. Remove trials
%%%%%%%%%%%%%%
EEG = pop_selectevent(EEG, 'epoch',[1:2] ,'deleteevents','on','deleteepochs','on','invertepochs','off');

%%%%%%%%%%%%%%
% 5. Average reference excluding eye electrodes
%%%%%%%%%%%%%%
EEG = pop_reref(EEG, [], 'exclude', [63 64]);

%%%%%%%%%%%%%%
% 6. ICA
%%%%%%%%%%%%%%
EEG = pop_runica(EEG, 'extended',1,'interupt','on', 'chanind', [1:64]);
%Save
EEG = pop_saveset(EEG);

%%%%%%%%%%%%%%  
% 7. Manually remove trials                                         
%%%%%%%%%%%%%%                                           
% Keep track of changes                                                                                                                                         
file_name = sprintf(strcat(currentParticipant...                                                
    ,recordingEnding,'_guide.txt'),1);                                             
dlmwrite(file_name, 1);                                                      
open(file_name);                                                              

%              <<<<-------------------------------------------------------%                                         
%                                                                         | 
EEG = pop_eegplot(EEG);%                                                  |
                       %                                                  |
%%%%%%%%%%%%%%                                                            |
% 8. Manually interpolate missing channels                                |
%%%%%%%%%%%%%%                                                            |
EEG = pop_interp(EEG);%                                                   |
                      %                                                   |
%%%%%%%%%%%%%%                                                            |
% 9. Manually highlight ICAs                                              |
%%%%%%%%%%%%%%                                                            |
EEG = pop_selectcomps(EEG);%                                              |
                           %                                              |
EEG = pop_subcomp(EEG);    %                                              |
                           %                                              |
%Repeat 7 8 9 until it looks good subjectively + log changes              |
%              -------------------------------------------------------->>>>
%Save at the end
EEG = pop_saveset(EEG);  

%%%%%%%%%%%%%%
% 10. %baseline correct if needed 
%%%%%%%%%%%%%%
%EEG = pop_rmbase(EEG, [-500 0]);

%%%%%%%%%%%%%%
% MANUAL MENU
%%%%%%%%%%%%%%
%See interface
eeglab redraw;
%Save
EEG = pop_saveset(EEG);