function [trainInput,trainTarget,testInput,testTarget] = distribution(inputSignal,desireddata,inputDimension,trainSize,testSize)
%Input training signal with data embedding
trainInput=zeros(inputDimension,trainSize);
for k = 1:trainSize
    if k < inputDimension
        trainInput(:,k) = [inputSignal(k:-1:1)';zeros(inputDimension-k,1)];
    else
        trainInput(:,k) =inputSignal(k:-1:k-inputDimension+1);
    end
end
trainTarget=zeros(trainSize,1);
for k = 1:trainSize
    trainTarget(k)=desireddata(k);
end
testInput=zeros(inputDimension,testSize);
for k = 1:testSize
    testInput(:,k) =inputSignal(k+trainSize:-1:k+trainSize-inputDimension+1);
end
testTarget=zeros(testSize,1);
for k = 1:testSize
    testTarget(k)=desireddata(k+trainSize);
end
end
