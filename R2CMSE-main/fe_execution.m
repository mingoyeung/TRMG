load("imf.mat");
[x,y]=size(imf);
m = 2;
r = 0.25;
n = 3;
fe=zeros(1,x);
for i=1:x
    fe(i) = func_FE_FuzzEn(imf(i,:),m,r,n)
end