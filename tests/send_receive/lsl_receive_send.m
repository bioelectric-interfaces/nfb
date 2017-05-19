%% instantiate the library
disp('Loading the library...');
lib = lsl_loadlib();

% resolve a stream...
disp('Resolving an EEG stream...');
result = {};
while isempty(result)
    result = lsl_resolve_byprop(lib,'name','python'); end

% create a new inlet
disp('Opening an inlet...');
inlet = lsl_inlet(result{1});

info = lsl_streaminfo(lib,'matlab','EEG', 5);

disp('Opening an outlet...');
outlet = lsl_outlet(info);
disp('Opened');
while true
    % get data from the inlet
    % disp('Now receiving data...');
    [vec,ts] = inlet.pull_sample();
    outlet.push_sample([2, 3, 4, 5, 6])
end