clear all,
close all
clc
%% data
mu_vec = linspace( 1e-4, 1e-2,3 );
length_mu_vec = length(mu_vec);
[ mse,thy ] = deal(zeros(length_mu_vec,1));
% Monter Carlo
MC = 50;
% Data section
[ inputDimension, trainSize, testSize ] = deal( 10, 1e5, 1e2 );
[ input_var, noise_var ] = deal( 1, 1e-3 );
generatenumber = inputDimension*trainSize;
h = waitbar( 0,'Curves are generating!' );
%% algorithm
% 选择不同的步长
for i = 1:length_mu_vec
    stepSizeWeightVector = mu_vec(i);
    Sum_learningCurve_LMMN = 0;
    % MC模拟仿真
    for mc = 1:MC
        display(mc)
        % 数据源
        Filter_w = (1/sqrt(inputDimension))*ones(1,inputDimension);
        out_first = 1;
        Inputsignal = sqrt(input_var)*randn(1,generatenumber);
        desired_sig_cle = filter(Filter_w,out_first,Inputsignal);
        % 噪声源
        Noise = sqrt(noise_var)*randn(1,length(desired_sig_cle));%because the size of inputsignal is equal the size of disired_sig_cle
        desired_sig_noise = desired_sig_cle+Noise;
        % Function section
        tic;
        [trainInput,trainTarget,testInput,testTarget] = distribution(Inputsignal,desired_sig_noise,inputDimension,trainSize,testSize);
        [ learningCurve, MSE1, MSE2 ]= deal(zeros(trainSize,1));
        weightVector = zeros(inputDimension,1);
        % training
        for n = 1:trainSize    
            Err = trainTarget(n) -  weightVector'*trainInput(:,n);
            weightVector = weightVector + stepSizeWeightVector*Err*trainInput(:,n);
            aprioriErr = (Filter_w - weightVector)'*trainInput(:,n);
            learningCurve(n) = mean(aprioriErr.^2);
       end
        Sum_learningCurve_LMMN = Sum_learningCurve_LMMN+learningCurve;
       toc;
    end
    %% plot
    Aver_learningCurve_LMMN = Sum_learningCurve_LMMN/MC;
   %figure section
    figure(1)
    plot(10*log10(Aver_learningCurve_LMMN),'-.r','LineWidth',2);hold on;
    mse(i) = mean(Aver_learningCurve_LMMN(end-200:end));
    thy(i) = (mu_vec(i)*inputDimension*input_var*noise_var)/(2 - mu_vec(i)*( inputDimension + 2 )*input_var);%large u;
    waitbar( i/length_mu_vec)
end
close(h)
%% plot
%figure section
figure(2)
plot(mu_vec,mse,'r:<','LineWidth',2);hold on;
plot(mu_vec,thy,'b-o','LineWidth',2);hold on;
% axis
xlabel('iteration','FontName','Times New Roman','FontSize',20); 
ylabel('EMSE','FontName','Times New Roman','FontSize',20); 
set(gca,'FontSize',18);
legend('Simulation','Theory')
title(' Experimental and theoretical EMSE versus \mu for LMS');