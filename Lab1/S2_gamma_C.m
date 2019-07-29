% Digital Forensics
% A.A. 2017/2018
% Lab. experience n.1 - Network forensics
% teacher: Simone Milani (simone.milani@dei.unipd.it)

clc; close all; clear global; clearvars;

% Load and visualize data

% Load features
load Scenario2_feat; % contains: fb_train, fb_test, tw_train, tw_test

% Write .mat files
write_svm_file(fb_train,tw_train,'train.mat');
write_svm_file(fb_test,tw_test,'test.mat');

tr_size = length(fb_train) + length(tw_train);

% Define label vectors
label_train = [ ones(size(fb_train,1),1) ; -1*ones(size(tw_train,1),1) ] ;
label_test = [ ones(size(fb_test,1),1) ; -1*ones(size(tw_test,1),1) ] ;

% Plot training data
figure();
plot(fb_train(:,1),fb_train(:,2),'b.', tw_train(:,1),tw_train(:,2),'ro');
xlabel('Avg. packet length (bytes)');
ylabel('Variance');
title('Training data'); grid on;
% xl1 = xlim; yl1 = ylim;

g_vec = 0:0.0001:0.0025;
c_vec = 1:0.5:5;
acc_vec = zeros(length(g_vec),length(c_vec));
c_acc = 0;
tStart = tic;

for i=1:length(g_vec)
    for j=1:length(c_vec)
        c = c_vec(j);
        g = g_vec(i);
        % Train libsvm classifier
        [~,r] = system(sprintf('svm-train.exe -c %f -t 2 -g %f -v 5 train.mat classifier.mod', c, g));
        k = strfind(r,'Cross Validation Accuracy = ');
        k_2 = strfind(r,'%');
        acc = str2double(r(k+28:k_2-1));
        c_acc = acc;
        [~,~] = system(sprintf('svm-train.exe -t 2 -g %f train.mat classifier.mod', c, g));
        [~,nbf,~,~] = read_libsvm_rbf('classifier.mod');
        acc_vec(i,j) = 1-acc/100 + 1.5*nbf/tr_size;
    end
end

[best, idx_best] = min(acc_vec(:));
[idx_g, idx_c] = ind2sub(size(acc_vec), idx_best);

best_g = g_vec(idx_g);
best_c = c_vec(idx_c);
toc(tStart);

fprintf('Cross Validation Accuracy : %.3f %\n', c_acc);
fprintf('\n');

figure, mesh(c_vec, g_vec, acc_vec);
hold on;
plot3(best_c, best_g, acc_vec(idx_g, idx_c), 'rx');
title('Validation error'); view(-140, 35);
xlim([1 5]); ylim([0 0.0025]); zlim([0 0.5]);
xlabel('C'), ylabel('Gamma'), zlabel('Loss %')

% figure()
% plot(g_vec,acc_vec(:,1));
% title('Validation error and % SVs vs parameter \gamma');
% grid on;
% xlabel('Parameter: \gamma'); ylabel('%');
% hold on;
% plot(g_vec,acc_vec(:,2)/tr_size);
% legend('Validation error','Support vector percentage');
% hold off;
% 
% % Best trade-off between validation error and number of support vectors
% X = acc_vec(:,1)+ 1.5*acc_vec(:,2)/tr_size;		% Compute minimum of this curve
% [best, idx_best] = min(X);
% g_opt = g_vec(idx_best);
% 
% figure()
% plot(g_vec,X);
% grid on;
% title('Trade-off curve (Val. error + % SVs) vs parameter \gamma');
% xlabel('Parameter: \gamma'); ylabel('%');
% hold on;
% plot(g_opt, acc_vec(idx_best,1)+acc_vec(idx_best,2)/tr_size,'*');
% hold off;
% legend('Trade-off curve', 'Best \gamma');
% ylim([0 0.5]);
% 
% %%
system(sprintf('svm-train.exe -t 2 -c %f -g %f train.mat classifier.mod', best_c, best_g));

[vector,nbf,rho,gamma] = read_libsvm_rbf('classifier.mod');
% % Test libsvm classifier 
% if ispc % check whether we are using Windows or Linux/MAC
    system(sprintf('svm-predict.exe test.mat classifier.mod output.txt'));
% else
%     system(sprintf('./svm-predict test.mat classifier.mod output.txt'));
% end
% 
% % Read classification results
fp = fopen('output.txt','r');
val_test = fscanf(fp,'%f',size(fb_test,1)+size(tw_test,1));
fclose(fp);
% 
% % Compute the accuracy
accuracy = sum(label_test==val_test)/length(val_test);
fprintf('Accuracy on test set: %.3f %\n',accuracy*100);
fprintf('\n');
% 
% %% Visualize results
% 
% Plot test data
% 
% Visualize support vector
hold on;
plot(vector(:,2),vector(:,3),'k*');
hold off;
% 
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
hold on; grid on;
plot(X0(iii), X1(iii), 'k.');
% axis([xl1 yl1]);