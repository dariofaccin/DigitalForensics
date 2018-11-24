clc; close all; clear global; clearvars;

% Load and visualize data

% Load features
load Scenario2_feat; % contains: fb_train, fb_test, tw_train, tw_test

% Write .mat files
write_svm_file(fb_train,tw_train,'train.mat');
write_svm_file(fb_test,tw_test,'test.mat');

tr_size = length(fb_train) + length(tw_train);

figure();
plot(fb_train(:,1),fb_train(:,2),'b.', tw_train(:,1),tw_train(:,2),'ro');
% plot(tw_train(:,1),tw_train(:,2),'ro');
grid on; hold on;
xlabel('Avg. packet length (bytes)');
ylabel('Variance');
title('Training data');
hold off;

tw_train = tw_train - mean(tw_train);
fb_train = fb_train - mean(fb_train);

tw_up = tw_train(:,1).^2 + tw_train(:,2).^2;
fb_up = fb_train(:,1).^2 + fb_train(:,2).^2;

figure();
scatter3(tw_train(:,1),tw_train(:,2),tw_up, 'r.');
hold on;
scatter3(fb_train(:,1),fb_train(:,2),fb_up, 'b.');