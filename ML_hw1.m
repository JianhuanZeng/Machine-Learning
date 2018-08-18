x = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\X_train.csv');
%X=transpose(X);
y = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\y_train.csv');

df = zeros(1,5001);
w = zeros(7,5001);
for i=0:5000
    [w(:,i+1),df(i+1)]=RiReg(x,y,i);
end



%-------------Question a----------------------
plot(df,w)
grid on
title('Problem 2.a')
xlabel('df(numbda)')
ylabel('w')
legend('d1','d2','d3','d4','d5','d6','d7')



%-------------Question c----------------------
xt = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\X_test.csv');
yt = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\y_test.csv');
[n,d]=size(xt);

yt=repmat(yt,1,501); %Repeat copies of arraycollapse
yp=xt*w(:,[1:501]);
rmse=(yt-yp).^2;
rmse=(sum(rmse)./n).^0.5;
% rmse2 = zeros(n,51);
% for i=1:51
%    rmse2(:,i)= (yt-Xt*w(:,i)).^2;
% end  

numbda=[0:500];
plot(numbda([1:51]),rmse([1:51]))
grid on
title('Problem 2.c')
xlabel('numbda')
ylabel('RMSE')



%-------------Question d----------------------
plot(numbda,rmse)
grid on
hold on
title('Problem 2.d')
xlabel('numbda')
ylabel('RMSE')
%-------
d=13;
x2=[x x(:,[1:6]).^2];
xt2=[xt xt(:,[1:6]).^2];
df2 = zeros(1,501);
w2 = zeros(d,501);
for i=0:500
    [w2(:,i+1),df2(i+1)]=RiReg(x2,y,i);
end
yp2=xt2*w2;
rmseb=(yt-yp2).^2;
rmseb=(sum(rmseb)./n).^0.5;
plot(numbda,rmseb)
%------
d=19;
x3=[x x(:,[1:6]).^2 x(:,[1:6]).^3];
xt3=[xt xt(:,[1:6]).^2 xt(:,[1:6]).^3];
df3 = zeros(1,501);
w3 = zeros(d,501);
for i=0:500
    [w3(:,i+1),df3(i+1)]=RiReg(x3,y,i);
end
yp3=xt3*w3;
rmsec=(yt-yp3).^2;
rmsec=(sum(rmsec)./n).^0.5;
plot(numbda,rmsec)
legend('order 1','order 2','order 3')