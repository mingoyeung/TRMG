tic
clc
clear all
%profile on
load("fimf.mat");%imf
[x,y]=size(imf);
tau = 2;
r = 0.15;
h = 2;
r2cmse=zeros(1,x);
for i=1:x
    r2cmse(i) = r2cmse_value(imf(i,:),r,h,tau)
end
%toc
%profileStruct = profile('info');
%[flopTotal, Details] = FLOPS('r2cmse_value', 'r2cmse_profile', profileStruct);



