% Digital Forensics
% A.A. 2017/2018
% Lab. experience n.1 - Network forensics
% teacher: Simone Milani (simone.milani@dei.unipd.it)

clc; close all; clear global; clearvars;

%% Load and visualize data

% Load features
load Scenario2_feat; % contains: fb_train, fb_test, tw_train, tw_test

% Write .mat files
write_svm_file(fb_train,tw_train,'train.mat');
write_svm_file(fb_test,tw_test,'test.mat');

% Define label vectors
label_train = [ ones(size(fb_train,1),1) ; -1*ones(size(tw_train,1),1) ] ;
label_test = [ ones(size(fb_test,1),1) ; -1*ones(size(tw_test,1),1) ] ;

% Plot training data
figure();
plot(fb_train(:,1),fb_train(:,2),'b.', tw_train(:,1),tw_train(:,2),'ro');
xlabel('Avg. packet length (bytes)');
ylabel('Variance');
title('Training data');
%xl1 = xlim; yl1 = ylim;

%% Train and test

% Train libsvm classifier
system(sprintf('svm-train.exe -t 2 -g 0.01 train.mat classifier.mod'));

% Read classifier data: Support vector
[vector,nbf,rho,gamma] = read_libsvm_rbf('classifier.mod');
	
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

% Visualize support vector
% hold on;
% plot(vector(:,2),vector(:,3),'k*');
% hold off;

min_x0 = min([fb_test(:,1); tw_test(:,1)]);
max_x0 = max([fb_test(:,1); tw_test(:,1)]);
min_x1 = min([fb_test(:,2); tw_test(:,2)]);
max_x1 = max([fb_test(:,2); tw_test(:,2)]);

[X0, X1] = meshgrid(linspace(min_x0,max_x0,1000),linspace(min_x1,max_x1,200));

diff_sv_vet=zeros(nbf,length(X0(:)));
for isv=1:nbf
	diff_sv_vet(isv,:)=exp(-1*gamma*mean((ones(length(X0(:)),1)* ...
	vector(isv,2:3)-[X0(:) X1(:)]).^2,2));
	diff_sv_vet(isv,:)=vector(isv,1)*diff_sv_vet(isv,:);
end
val_line=sum(diff_sv_vet,1)-rho; %line points
iii=find(abs(val_line)<0.05); %for line points coordinates are close to 0

%% Plot results
figure();
plot(fb_test(:,1),fb_test(:,2),'b.', tw_test(:,1),tw_test(:,2),'ro');
xlabel('Avg. packet length (bytes)');
ylabel('Variance');
title('Test data');
hold on;
plot(X0(iii), X1(iii), 'k.');
%axis([xl1 yl1]);