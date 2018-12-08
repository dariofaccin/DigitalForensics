% Digital Forensics
% A.A. 2018/2019
% Lab. experience n.2 - Camera ballistics 
% teacher: Simone Milani (simone.milani@dei.unipd.it)
set(0,'DefaultTextInterpreter','latex');

clc, clear, close all
addpath('./filter')
addpath('./functions')


%% Input directories and parameters

%%% Set directories of flatfield and natural images 
flat_img_dir = './img_flat';
nat_img_dir  = './img_nat_comp';

%%% Set size of the image portion to be analyzed (in pixels)
window_size = 512; % try 256, 512, 1024, 2048


%% Load images

%%% Flatfield images
flat_img_list = dir(fullfile(flat_img_dir, '*.jpg'));
num_flat = length(flat_img_list);
for i = 1:num_flat
    I_flat{i} = imread(fullfile(flat_img_dir, flat_img_list(i).name)); %#ok<*SAGROW>
    I_flat{i} = I_flat{i}(1:window_size,1:window_size,:);
end

%%% Natural images
nat_img_list = dir(fullfile(nat_img_dir, '*.jpg'));
num_nat = length(nat_img_list);
for i = 1:num_nat
    I_nat{i} = imread(fullfile(nat_img_dir, nat_img_list(i).name));
    I_nat{i} = I_nat{i}(1:window_size,1:window_size,:);
end


%% Compute noise for each image

%%% Flatfield images
for i = 1:num_flat
    W_flat{i} = NoiseExtract(I_flat{i}, MakeONFilter('Daubechies',8), 3, 4);
end

%%% Natural images
for i = 1:num_nat
    W_nat{i} = NoiseExtract(I_nat{i}, MakeONFilter('Daubechies',8), 3, 4);
end


%% Compute PRNU from flatfield images

%%%  Initialize variables
[M,N,~]=size(I_flat{1});
for j=1:3
    RPsum{j}=zeros(M,N,'single');
    NN{j}=zeros(M,N,'single'); % number of additions to each pixel for RPsum
end

%%% ML estimator
for i = 1:num_flat % for each image
    X = double255(I_flat{i});
    for j=1:3 % for each RGB channel
        ImNoise = single(W_flat{i}(:,:,j));
        Inten = single(IntenScale(X(:,:,j))).*Saturation(X(:,:,j));    % zeros for saturated pixels
        RPsum{j} = RPsum{j}+ImNoise.*Inten;   	% weighted average of ImNoise (weighted by Inten)
        NN{j} = NN{j} + Inten.^2;
    end
end

RP = cat(3, RPsum{1}./(NN{1}+1), RPsum{2}./(NN{2}+1), RPsum{3}./(NN{3}+1));

%%% PRNU post processing
RP = ZeroMeanTotal(RP); % Remove linear pattern
RP = single(RP);
RP = rgb2gray1(RP);
K = WienerInDFT(RP, std2(RP));


%% Compute correlation

for i = 1:num_nat
    
    %%% Noise post processing
    W_nat_proc = rgb2gray1(W_nat{i});
    W_nat_proc = ZeroMeanTotal(W_nat_proc);
    W_nat_proc = single(W_nat_proc);
    W_nat_proc = WienerInDFT(W_nat_proc, std2(W_nat_proc));
    
    %%% Correlation
    Ix = double(rgb2gray(I_nat{i}));
    C = crosscorr(W_nat_proc, Ix.*K);
    detection{i} = PCE(C);
    
end


%% Display results

for i = 1:num_nat
    disp(nat_img_list(i).name)
    detection{i} %#ok<*NOPTS>
end

data_vect = zeros(num_nat,2);
for i=1:num_nat
	data_vect(i,1) = detection{1,i}.PCE;
	data_vect(i,2) = detection{1,i}.P_FA;
end

figure()
labels = strsplit(num2str(1:num_nat));
semilogy(data_vect(:,1),data_vect(:,2),'o'); grid on;
text(data_vect(:,1),data_vect(:,2),labels,'VerticalAlignment','bottom','HorizontalAlignment','left');
title('PFA and PCE versus Compression Ratio');
xlabel('$PCE$'); ylabel('log $P_{FA}$');

figure()
labels = strsplit(num2str(1:num_nat));
plot(data_vect(:,1),data_vect(:,2),'o'); grid on;
text(data_vect(:,1),data_vect(:,2),labels,'VerticalAlignment','bottom','HorizontalAlignment','left');
title('PFA and PCE versus Compression Ratio');
xlabel('$PCE$'); ylabel('$P_{FA}$');