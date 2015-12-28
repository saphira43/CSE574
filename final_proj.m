function []=proj3_main()
images1=loadMNISTImages('train-images.idx3-ubyte');
labels1=loadMNISTLabels('train-labels.idx1-ubyte');
labels_tk=zeros(60000,10);

for j=1:60000
labels_tk(j,labels1(j,1)+1)=1;
end

[Wlr,blr,error]=gradient_descent(images1,labels_tk,labels1);
M=15;
[Wnn1,Wnn2,bnn1,bnn2,error_NN]=NN_test(images1,labels_tk,labels1,M);

save('proj3.mat');
end
 
function [Wlr,blr,error]=gradient_descent(images1,labels_tk,labels1)

Wlr=zeros(784,10);
blr=zeros(1,10);
img=transpose(images1);
eta=0.003;
r=1;
error=zeros(21,1);
plt_y=zeros(21,1);
test=0;
 y_fin=zeros(60000,10);
 classify=zeros(60000,1);
 max_class=zeros(60000,1);
for tot=1:20
for j=1:60000
    denom=0;
    for i=1:10
        denom=denom+exp(img(j,:)*Wlr(:,i)+blr(1,i));
    end
    for k=1:10
        y=exp(img(j,:)*Wlr(:,k)+blr(1,k));
        y_fin(j,k)=y/denom;
        grad=eta*(y_fin(j,k)-labels_tk(j,k))*images1(:,j);
        Wlr(:,k)=Wlr(:,k)-grad;
    end
end
prev_test=test;
test=0;
for j=1:60000
   classify(j)=y_fin(j,1);
   error(r)=error(r)+(labels_tk(j,1)*log(y_fin(j,1)));
   for k=2:10
       error(r)=error(r)+(labels_tk(j,k)*log(y_fin(j,k)));
       if y_fin(j,k)>classify(j)
            classify(j)=y_fin(j,k);
            max_class(j)=k-1;
       end
   end
   if (max_class(j)~=labels1(j))
    test=test+1;
   end
end
error(r)=-1*error(r);
plt_y(r)=60000*r;
r=r+1;

if(test<prev_test)
    eta=eta*1.3;
else
    eta=eta*0.4;
end
end
plot(plt_y,error);
end

function [Wnn1,Wnn2,bnn1,bnn2,error]=NN_test(images1,labels_tk,labels1,M)
%Weights for layer 1
a=-1*(1/28);
b=1/28;
Wnn1=(b-a)*rand(784,M)+a;

%weights for layer 2
a1=-1*(1/4);
b1=1/4;
Wnn2=(b1-a1)*rand(M,10)+a1;
bnn1=zeros(1,M);
bnn2=zeros(1,10);

img=transpose(images1);

eta=0.1;

error=zeros(250,1);
plt_y=zeros(250,1);
test=0;

y_fin=zeros(60000,10);
classify=zeros(60000,1);
max_class=zeros(60000,1);
%z=zeros(M,1);
%delta_k=zeros(10,1);
%delta_j=zeros(M,1)

for test_runs=1:250

for j=1:60000
    for k=1:M
        aj=(img(j,:)*Wnn1(:,k));
        %Value of input to next layer using tanh function
        z(k,1)=tanh(aj);
    end
    
    %calculting denom for yk.
    denom=0;
    for i=1:10
    denom=denom+exp(transpose(z(:,1))*Wnn2(:,i));
    end
    
    
    %Putting it in final layer
    for r=1:10
    numer=transpose(z(:,1))*Wnn2(:,r);
    y_fin(j,r)=exp(numer);
    y_fin(j,r)=y_fin(j,r)/denom;
    delta_k(r,1)=y_fin(j,r)-labels_tk(j,r);
    end
    
    
    %back propogating to first layer
    for u=1:M
        try1=0;
        for v=1:10
            %putting old Wnn2,can change
            try1=try1+Wnn2(u,v)*delta_k(v,1);
        end
        delta_j(u,1)=(1-(z(u,1)*z(u,1)))*try1;
    end
    
    %applying gradient for second layer
    for s=1:M
        for t=1:10
            grad=delta_k(t,1)*z(s,1);
            Wnn2(s,t)=Wnn2(s,t)-(eta*grad);
        end
    end
    
    %applying gradient for first layer
    for xr=1:M
        for mid=1:784
            grad=delta_j(xr,1)*(images1(mid,j));
            Wnn1(mid,xr)=Wnn1(mid,xr)-(eta*grad);
        end
    end
end
%error=0;
prev_test=test;
test=0;
for j=1:60000
   classify(j)=y_fin(j,1);
   error(test_runs)=error(test_runs)+(labels_tk(j,1)*log(y_fin(j,1)));
   for k=2:10
       if y_fin(j,k)>classify(j)
            error(test_runs)=error(test_runs)+(labels_tk(j,k)*log(y_fin(j,k)));
            classify(j)=y_fin(j,k);
            max_class(j)=k-1;
       end
   end
   if (max_class(j)~=labels1(j))
    test=test+1;
   end
end
if(test>prev_test)
    eta=eta*0.4;
else
    eta=eta*1.2;
end
error(test_runs)=-1*error(test_runs);
plt_y(test_runs)=test_runs*60000;
end
end

function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
%images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end

function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end


