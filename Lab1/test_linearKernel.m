% Digital Forensics
% A.A. 2017/2018
% Lab. experience n.1 - Network forensics
% teacher: Simone Milani (simone.milani@dei.unipd.it)

clear
close all


%% Load and visualize data

% Load features
load Scenario1_feat; % contains: fb_train, fb_test, tw_train, tw_test

% Write .mat files
write_svm_file(fb_train,tw_train,'train.mat');
write_svm_file(fb_test,tw_test,'test.mat');

% Define label vectors
label_train = [ ones(size(fb_train,1),1) ; -1*ones(size(tw_train,1),1) ] ;
label_test = [ ones(size(fb_test,1),1) ; -1*ones(size(tw_test,1),1) ] ;

% Plot training data
figure(1);
plot(fb_train(:,1),fb_train(:,2),'b.', tw_train(:,1),tw_train(:,2),'ro');
xlabel('Avg. packet length (bytes)');
ylabel('Variance');
title('Training data');
xl1 = xlim; yl1 = ylim;


%% Train and test

% Train libsvm classifier
if ispc % check whether we are using Windows or Linux/MAC
    system(sprintf('svm-train.exe -t 0 train.mat classifier.mod'));
else
    system(sprintf('./svm-train -t 0 train.mat classifier.mod'));
end

% Read classifier data: Support vector
[vector,nsv,rho] = read_libsvm_sv('classifier.mod');

% Test libsvm classifier 
if ispc % check whether we are using Windows or Linux/MAC
    system(sprintf('svm-predict.exe test.mat classifier.mod output.txt'));
else
    system(sprintf('./svm-predict test.mat classifier.mod output.txt'));
end

% Read classification results
fp = fopen('output.txt','r');
val_test = fscanf(fp,'%f',size(fb_test,1)+size(tw_test,1));
fclose(fp);

% Compute the accuracy
accuracy = sum(label_test==val_test)/length(val_test);
fprintf('Accuracy on test set: %.3f %\n',accuracy*100);
fprintf('\n');


%% Visualize results

% Plot test data
figure(2);
plot(fb_test(:,1),fb_test(:,2),'b.', tw_test(:,1),tw_test(:,2),'ro');
xlabel('Avg. packet length (bytes)');
ylabel('Variance');
title('Test data');

% Visualize support vector
hold on;
plot(vector(:,2),vector(:,3),'k*');
hold off;

% Compute line parameters
w = sum(vector(:,1).*vector(:,2:3));

% Compute slope-intercept form
m = w(1)/w(2); % slope
q = rho/w(2);  % intercept

% Plot the line
figure(2);
hold on;
fplot(@(x) q - m*x, yl1, 'k-', 'LineWidth', 1);
axis([xl1 yl1]);

