%% multi-image data driven tight frame for cryo-EM denoising when SNR=0.1
clear all;close all;clc
addpath('../tool/');
NN=20; % use 20 images to learn filters.

%% Parameter setting of image denoising
sigma     	= 45;                            			% noise level
patchSize 	= 16; 										% patch size
stepSize  	= 1;                       					% overlap step of data   
% trainnum	= 225*NN;									% the number of samples selected for learning
trainnum	= 12769*NN;									% the number of samples selected for learning
lambda_1  	= 3.5 * sigma;            					% lambda for learning dictionary
lambda_2  	= 2.7 * sigma;            					% lambda for denoising by learned dictionary
opts.nIter	= 90;										% number loop for constructing data-driven filter bank
opts.A 		= (1/patchSize)*ones(patchSize^2,1);		% pre-input filters  (must be orthogonal)

%% Generate simulated noisy image
randn('seed',2013); rand('seed',2013)

fileName1  	= 'clean/000041@center_str3_align_128pixel_50000.png'; % clean image
clean_img = double(imread(fileName1));
kk=uint8(clean_img);
% noisy_img 	= double(imread(fileName)); % read image   
% [h, w] 	  	= size(noisy_img);  % image size, 128*128

devset = 'list_snr01.txt'; %list_snr01
datapath='./SNR0.1';
d_files=textread(devset,'%s');

for ii=1:length(d_files)
    d_files_1{ii} = fullfile(datapath, d_files{ii});     
end

n_img=cell(1,NN);
for ii=1:NN
    files=d_files_1{ii};
    noisy_img_1 = double(imread(files));
    % regularization
    maxPixelValue = max(max(noisy_img_1)); % find the maximum
    minPixelValue = min(min(noisy_img_1)); % find the minimum
    % make it to the range between 0 to 1 and then multiply by 255.
    noisy_img1 = ((noisy_img_1 + minPixelValue)/maxPixelValue)*255; % to 0-255 scale
%     noisy_img2 = (noisy_img + minPixelValue)/maxPixelValue; % normalize to 0-1 scale
    n_img{ii}=noisy_img1;
end

% convert image scale
maxPixelValue1 = max(max(clean_img)); % find the maximum
minPixelValue1 = min(min(clean_img)); % find the minimum
% make it to the range between 0 to 1 and then multiply by 255.
clean_img1 = ((clean_img + minPixelValue1)/maxPixelValue1)*255; % to 0-255 scale
% clean_img2 = (clean_img + minPixelValue1)/maxPixelValue1; % normalize to 0-1 scale
tic;
%% Checking the correctness of pre-defined filter subset A 
A = opts.A;
if ~isempty(A)
	r = size(A, 2);
	temp = wthresh(A'*A - eye(r),'h',1e-14);
	if sum(temp(:)) > 0 
		error('The input A does not meet the requirement!');
	end
end

%% Generate collection of image patches
Data_1=[];
for ii=1:NN
    Data  		= im2colstep(n_img{ii}, [patchSize, patchSize], [stepSize, stepSize]);
    Data_1=[Data_1 Data];
end
rperm 		= randperm(size(Data_1, 2));
% patchData 	= Data_1(:, rperm(1:trainnum));
patchData 	= Data_1(:, rperm(1:trainnum));

%% Learning filter bank from image patches
learnt_dict  = filter_learning(patchData, lambda_1, opts);

%% Denoising image by using the tight frame derived from learned filter banks
% for one image
im_out 		 = frame_denoising(n_img{1}, learnt_dict, lambda_2);
figure; imshow(im_out,[])

PSNRoutput 	 = Psnr(clean_img1, round(im_out));
PSNRinput = Psnr(clean_img1, noisy_img1);
err=clean_img1-round(im_out);
mse_1=sum(err(:)).^2/numel(err);

% for the whole image
Img=[];
for ii=1:NN
    Im=n_img{ii};
    Img=[Img Im];
end
Im_out 		 = frame_denoising(Img, learnt_dict, lambda_2);
Im_out_1=Im_out(:,1:128);
figure; imshow(Im_out_1,[])
PSNRoutput_1 	 = Psnr(clean_img1, round(Im_out_1));

save('str3_001_000041_20p_m1.mat','im_out') % denoise one image with learned dictionary
save('str3_001_000041_20p_m2.mat','Im_out_1') % denoise the whole images with learned dictionary 


toc;
