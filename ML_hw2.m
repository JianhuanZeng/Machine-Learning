%% Problem 2a: NAIVE BAYES CLASSIFIER %%

% Read Data
x = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\hw2-data\X_train.csv');
y = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\hw2-data\y_train.csv');
[n,d] = size(x);

% Compute Pi
n1=sum(y(:) ==1); % compute the amount of spam for trian data
n0=sum(y(:) ==0); % compute the amount of non_spam for trian data
pi=n1/n;

% Compute theta_1 and theta_2
a1=sum(x([1:1776],[1:54]))/n1;
a0=sum(x([1777:n],[1:54]))/n0;
b1=n1./sum(log(x([1:1776],[55:d])));
b0=n0./sum(log(x([1777:n],[55:d])));

% Classifier with loading X_test(x0)
xo = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\hw2-data\X_test.csv');
yr = csvread('C:\Users\joy28\Documents\Spring 2018 Study\ENEL4903ML\hw2-data\y_test.csv');
% compute yp0 %lnp0=zeros(93,57);
for j=1:54
    lnp0(:,j)=xo(:,j).*log(a0(j))+(1-xo(:,j)).*log(1-a0(j));
end

for j=1:3
    lnp0(:,j+54)=log(b0(j))-(1+b0(j)).*log(xo(:,j+54));
end
lnp0=sum(lnp0,2)+log(pi);

% compute yp1
for j=1:54
    lnp1(:,j)=xo(:,j).*log(a1(j))+(1-xo(:,j)).*log(1-a1(j));
end

for j=1:3
    lnp1(:,j+54)=log(b1(j))-(1+b1(j)).*log(xo(:,j+54));
end
lnp1=sum(lnp1,2)+log(1-pi);
% derive yp
yp=lnp0-lnp1;
for i=1:93
    if yp(i)>=0
        yp(i)=0;
    else
        yp(i)=1;
    end
end
% compare
yr1=sum(yr(:) ==1);
yr0=sum(yr(:) ==0);
yp1=sum(yp(:) ==1);
yp0=sum(yp(:) ==0);



%% -------------------------------------------------
%% Problem 2b : Stem Plot of theta_1
stem(a0)
hold on
stem(a1)
title('Bernoulli parameters')
xlabel('dimension')
legend('class0','class1')



%% -------------------------------------------------
%% Problem 2c : k-NN Algorithm  %%
function [ class ] = knn( u,x,y )
[n,d] = size(x);
u=repmat(u,n,1);
d=sum(abs(u-x),2);
% d = maxk(d,20); finding the largest 20 value in d
for k=1:20
    [mink, indk] = min(d);
    %max(k)=max;
    class(k)=y(indk);
    d(indk)=max(d);
end

for i=1:93
    u=xo(i,:);
    class(i,:)=knn(u,x,y );
    for k=1:20
        yp(i,k)=sum(class(i,[1:k]));
        if yp(i,k)/k>0.5
            yp(i,k)=1;
        else
            yp(i,k)=0;
        end
    end
end
yr=repmat(yr,1,20);
e=yr-yp;
for k=1:20
    accuracy(k)=sum(e(:,k) ==0)/93;
end
plot(accuracy)
title('the prediction accuracy of K-nn Algorithm')
xlabel('k')



%% -------------------------------------------------
%% Problem 2d : the steepest ascent algorithm %%
x=[ones(4508,1) x];
w=zeros(1,58);
for t=1:10000
    s=0;
    Sn=zeros(1,58);
    for i=1:4508
        tmp1=(2*y(i)-1)*x(i,:)*transpose(w);
        if tmp1<-12
            sig=0;
            s=s+tmp1;
        elseif tmp1>12
            sig=1;
            s=s;
        else
            sig=1/(exp(-tmp1)+1);
            s=s+log(sig);
        end
        Sn=Sn+(1-sig)*(2*y(i)-1)*x(i,:);% sum of dL
    end
    w=w+Sn/sqrt(t+1)/100000;
    L(t)=s;
end
plot(L)
title('L of the steepest ascent algorithm')
xlabel('iteration')


%% -------------------------------------------------
%% Problem 2e : the Newton Method %%
%x=[ones(4508,1) x];
w=zeros(1,58);
for t=1:100
    s=0;
    Sn=zeros(1,58);
    Sn2=zeros(58);
    for i=1:4508
        tmp2=x(i,:)*transpose(w);
        tmp1=(2*y(i)-1)*tmp2;
        if tmp1<-12
            sig=0;
            s=s+tmp1;
        elseif tmp1>12
            sig=1;
            s=s;
        else
            sig=1/(exp(-tmp1)+1);
            s=s+log(sig);
        end
        Sn=Sn+(1-sig)*(2*y(i)-1)*x(i,:);% sum of d(L)

        if abs(tmp2)>12
            Sn2=Sn2;
        else
            sig=1/(exp(-tmp1)+1);
            Sn2=Sn2+sig*(1-sig)*transpose(x(i,:))*x(i,:); %sum of d(d(L))
        end
    end
    w=w+Sn*transpose(inv(Sn2))/sqrt(t+1);
    L(t)=s;
end
plot(L)
title('L in the Newtons Method')
xlabel('iteration')

xo=[ones(93,1) xo];
yp=sign(xo*transpose(w));
% accuracy
e=(2.*yr-1)-yp;
accuracy=sum(e(:) ==0)/93;
