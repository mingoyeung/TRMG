tic
%profile on
load("fujiandata.mat");%data
sig=origin_fujian(:,1);%origin_fujian,data
fs = 1; 
t = 1:1:11520;%2880
[imf,ort,nbits] = emd(sig);
%toc
%profileStruct = profile('info');
%[flopTotal, Details] = FLOPS('emd', 'emd_profile', profileStruct);