%tic
%profile on
load("fujian.mat");%fujian.mat
imf=tvf_emd(Power);%DE_tennet_wind_offshore_generation_actual
%toc
%profileStruct = profile('info');
%[flopTotal, Details] = FLOPS('tvf_emd', 'tvf_emd_profile', profileStruct);
