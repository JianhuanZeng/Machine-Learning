
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ K ] = mkernel( x1,x,b )
    % gaussian kernel of a vector x1 and xi from a matrix
    % x1 is a vector
    % x is a matrix, consisting of n vectors xi
    % b is a scalar used in guassian kernel

    [n,d]=size(x);
    for i=1:n
        % distance between x1 and xi
        dis=sum(((x1-x(i,:)).^2));
        % gaussian kernel
        K(i)=exp(-dis/b);
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ yp, rmse ] = kgp_predict( xt,x,b,var )
    % a gaussian process prediction
    % x is a n*d training matrix, consisting of n training vectors xi
    % xt is a m*d testing matrix.
    % b is a scalar used in guassian kernel
    % var is a scalar used in guassian process prediction

    [n,d]=size(x);
    [m,d]=size(xt);

    % kernel of training data x.
    for i=1:n
        Kn(i,:)=mkernel(x(i,:),x,b);
    end
    % a value used to predict yp
    tmp=inv(var*eye(350)+Kn);

    % predict yp for m training vectors xi
    for i=1:m
      K=mkernel(xt(i,:),x,b);
      yp(i)=K*tmp*y;
    end

    % root mean square error
    tmp=(yt-transpose(yp)).^2;
    rmse=(sum(tmp)./42).^0.5;
  end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% data loading
x = csvread('/Users/cengjianhuan/Documents/ML/gaussian_process/X_train.csv');
y = csvread('/Users/cengjianhuan/Documents/ML/gaussian_process/y_train.csv');
xt = csvread('/Users/cengjianhuan/Documents/ML/gaussian_process/X_test.csv');
yt = csvread('/Users/cengjianhuan/Documents/ML/gaussian_process/y_test.csv');
b=[5,7,9,11,13,15];
var=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
