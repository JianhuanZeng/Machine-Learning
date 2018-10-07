% ------------------------------- Ridge Regression --------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ w, df ] = RiReg( x,y,numbda )
    % Ridge Regression for input data (x,y) and the hyperparameter numbda
    % x is a matrix (n,d), consisting of n vector xi, each xi with d features
    % y is a vector (n,1) 
    % numbda λ is a scalar, the penalty of w
    
    [n,d]=size(x);
    
    [U,S,V] = svd(x);
    s = diag(S);
    
    s1 = zeros(d,1);
    s2 = zeros(d,1);
    for i=1:d
        s1(i)=s(i)/(numbda+s(i)^2);
        s2(i)=s(i)*s1(i);
    end
    
    w=V*diag(s1)*transpose(U(:,[1:d]))*y;
    df=sum(s2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\X_train.csv');
y = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\y_train.csv');



% ----------------------- Question a:plot w_RR as a function of df (λ) --------------------------
% d = 7, the number of features in xi
% numbda λ = 0,1,2,3,...,5000

df = zeros(1,5001);
w = zeros(7,5001);

for i=0:5000
    % i is the penalty of w
    [w(:,i+1),df(i+1)]=RiReg(x,y,i);
end

plot(df,w)
grid on
title('Problem 2.a')
xlabel('df(numbda)')
ylabel('w')
legend('d1','d2','d3','d4','d5','d6','d7')
% The 4th dimension (d4 = car weight) and 6th dimension ( d6= car year) clearly stand out over the other dimensions.


% ------------------------------ Question c: Plot the root mean squared error ----------------------
% For λ = 0, . . . , 50, predict all 42 test cases
xt = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\X_test.csv');
yt = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\y_test.csv');
[n,d]=size(xt);

yt=repmat(yt,1,501); %Repeat copies of array
yp=xt*w(:,[1:501]);
rmse=(yt-yp).^2;
rmse=(sum(rmse)./n).^0.5;

numbda=[0:500];
plot(numbda([1:51]),rmse([1:51]))
grid on
title('Problem 2.c')
xlabel('numbda')
ylabel('RMSE')



% ----------------- Question d: a pth-order polynomial regression model for p = 1, 2, 3 ----------------------
% ------- p=1, d=6+1 ------
plot(numbda,rmse)
grid on
hold on
title('Problem 2.d')
xlabel('numbda')
ylabel('RMSE')

% ------- p=2, d=2*6+1 ------
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
% ------ p=3, d=3*6+1 ------
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
