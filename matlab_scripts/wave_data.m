% This script creates a .txt file for the wavestaffs
addpath('C:\Users\yuksel.alkarem\Documents\MATLAB\f(x)');
clear;clcc;
lambda = 70;
ramp_time = 60;
cwd_dir = cd;
cd ../data/FOCAL_wavedata/
myfiles = dir('IR-*.csv');
for i = 1:length(myfiles)
    data = csvread(myfiles(i).name,6,0);
    DT = data(2,1) - data(1,1);
    t_2 = 0:DT:(length(data)-1).*DT;
    t = t_2';
    ramp_index = find(t==ramp_time);
    t = t(ramp_index:end)-ramp_time;
    probes = data(ramp_index:end,5:9);
    t = t.*sqrt(lambda);
    probes = probes.*lambda;
    writematrix([t, probes], strcat("IR_ready_", num2str(i), ".csv"));
    [~, H, ~, ~] = zero_up_crossing(t,probes(:,1));
    H_max(i) = max(H);
end
cd(cwd_dir)