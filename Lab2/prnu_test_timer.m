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
nat_img_dir  = './img_nat';

ws_vect = [16 32 64 128 256 512 1024 2048];
num_real = 20;
times_vect = zeros(length(ws_vect),num_real);

for w=1:length(ws_vect)
	for real=1:num_real
		%%% Set size of the image portion to be analyzed (in pixels)
		window_size = ws_vect(w);
		tic;
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
		times_vect(w,real) = toc;
	end
end

%% PLOT
times_vect_mean = mean(times_vect,2);

figure()
plot(ws_vect, times_vect_mean); grid on;
title('Time required versus windowsize');
xlabel('windowsize [pixels]'); ylabel('t [s]');
xlim([ws_vect(1) ws_vect(end)]);
xticks(ws_vect);
xticklabels({'16','','','128','256','512','1024','2048'});