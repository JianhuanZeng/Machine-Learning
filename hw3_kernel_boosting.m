%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Question 1: the Gaussian kernel model for regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data loading
x = csvread('/Users/cengjianhuan/Documents/ML/gaussian_process/X_train.csv');
y = csvread('/Users/cengjianhuan/Documents/ML/gaussian_process/y_train.csv');
xt = csvread('/Users/cengjianhuan/Documents/ML/gaussian_process/X_test.csv');
yt = csvread('/Users/cengjianhuan/Documents/ML/gaussian_process/y_test.csv');
b=[5,7,9,11,13,15]; % the hyperparameter in Gaussian kernel
var=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]; %  the hyperparameter for the variance of y vallue.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ K ] = mkernel( x1,x,b )
    % gaussian kernel of a vector from a matrix
    % x1 is a vector
    % x is a matrix, consisting of n vector xi
    % b is a scalar
    [n,d]=size(x);
    for i=1:n
        dis=sum(((x1-x(i,:)).^2));
        K(i)=exp(-dis/b);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k=1:6
  % Get the n*n kernel matrix for training data, n=350
  for i=1:350
      Kn(i,:)=mkernel(x(i,:),x,b(k));
  end

  % Predict yp for each var(j)
  for j=1:10
    tmp=inv(var(j)*eye(350)+Kn);
    % to avoid INF when inversing the matrix tmp
    % tmp=var(j)*eye(350)+Kn;
    % [U,S,V] = svd(tmp);
    % tmp=V*inv(S)*transpose(U);

    for i=1:42
      K=mkernel(xt(i,:),x,b(k));
      yp(i)=K*tmp*y;
    end

    % rmse
    tmp=(yt-transpose(yp)).^2;
    rmse(k,j)=(sum(tmp)./42).^0.5;
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x4=x(:,4);
for i=1:350
    Kn(i,:)=mkernel(x4(i),x4,5);
end
tmp=inv(2*eye(350)+Kn);
for i=1:350
  yp(i)=Kn(i,:)*tmp*y;
end
yp=transpose(yp);

m=[x4;yp];
sortrows(m)；
plot(ans(:,1),ans(:,2))
hold on
plot(x4,y,'x')
title('question 1.d')
xlabel('dimension 4')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Question 2: Boosting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = csvread('/Users/cengjianhuan/Documents/ML/boosting/X_train.csv');
y = csvread('/Users/cengjianhuan/Documents/ML/boosting/y_train.csv');
xt = csvread('/Users/cengjianhuan/Documents/ML/boosting/X_test.csv');
yt = csvread('/Users/cengjianhuan/Documents/ML/boosting/y_test.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ w, yp] = ls( x,y )
%ls Summary of this function goes here
  % a least square linear regression classifier
  % Get the 1*d weight vector w
  w=inv(x*transpose(x))*x*y;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n,d]=size(x);
[m,d]=size(xt);
% weight for each Bt
wt=ones(1,n)/n;
e=zeros(1,1500); %epsilon
alpha=zeros(1,1500);
times=zeros(1500,1);
ypp=zeros(1,m);
ytt=zeros(1,n);

for t=1:1500
    % pick a database Bt
    idx=gendist(wt,1,n);
    for i=1:n
      xb(i,:)=x(idx(i),:);
      yb(i)=y(idx(i));
      times(idx(i))=times(idx(i))+1;
    end

    % learn from the database Bt
    w(:,t) = ls(xb,transpose(yb));
    % epsilon and alpha
    for i=1:n
      train(i)=sign(x(i,:)*w(:,t));
      et1(i)=y(i)*train(i)
      if et(i)==-1
          e(t)=e(t)+wt(i);
      end
    end
    % if training error is too large
    if e(t)>0.5
      et=-et;
      e(t)=1-e(t);
    end
    % alpha
    alpha(t)=log((1-e(t))/e(t))/2;

    % Train
    for i=1:n
      ytt(i)=ytt(i)+alpha*train(i);
      yttt(i)=sign(ytt(i));
      et3(i)=yttt(i)*yt(i);
    end
    etrain(t)=sum(et3(:) ==-1)/n;

    % Predict
    yp=sign(xt*w);
    for i=1:m
      ypp(i)=ypp(i)+alpha*yp(i);
      yppp(i,t)=sign(ypp(i));
      et2(i)=yppp(i,t)*yt(i);
    end
    etest(t)=1-sum(et2(:) == 1)/m;

    % update distribution w
    for i=1:n
      wt(i)=wt(i)*exp(-et1(i)*alphat(t));
    end

    wt=wt./sum(wt);
end

plot(etrain)
hold on
plot(etest)
legend('train error','test error')


% part_b %%%%%%%%%%%%%%%%%%%%%%%
upper_bound = zeros(1500,1);
upper_bound(1) = (0.5 - e(1))^2;
for t = 2:1500
    upper_bound(t) = upper_bound(t - 1) + (0.5 - e(t)) ^ 2;
end
upper_bound = exp(-2 .* upper_bound);

plot(upper_bound);
title('the upper bound')

% part_c %%%%%%%%%%%%%%%%%%%%%%%
times = transpose(times);
bar (times)
xlabel('data point')
ylabel('times')

% part_d %%%%%%%%%%%%%%%%%%%%%%%
plot(epsilon)
title('epsilon')
xlabel('t')

plot(alpha)
title('alpha')
xlabel('t')
