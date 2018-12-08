clc; close all; clear global; clearvars;

load('times.mat');
times_mean = mean(times_vect,2);
f = fit(ws_vect.',times_mean,'poly2');

figure()
plot(f,ws_vect,times_mean,'x'); grid on;
title('Curve Fitting');
xlabel('windowsize [pixels]'); ylabel('t [s]');
xlim([ws_vect(1) ws_vect(end)]);
xticks(ws_vect);
xticklabels({'16','','','128','256','512','1024','2048'});