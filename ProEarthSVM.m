%Theodore Wiebold Final Project: Battle of SVMs
clc
clear all

shook = xlsread('ProEarth_TRAIN.xlsx');  %Training data set

Y = shook(:,1); %category
X = shook(:, 2:1:513); %data

crack = xlsread('ProEarth_TEST.xlsx'); %Test data set
newX = crack(:, 2:1:513); %Test data
newY = crack(:,1); %Test category

SMOSVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF',...
    'Solver', 'SMO'); %SMO SVM 

LSMO = loss(SMOSVMModel,newX,newY); %raw missclassification

yfit1 = predict(SMOSVMModel,newX); 
cm1 = confusionchart(newY,yfit1)
title('SMOSVMModel')

% [x1,y1,~,auc1] = perfcurve(newY,yfit1,1);
% plot(x1,y1)
% xlabel('False positive rate'); ylabel('True positive rate');
% title('ROC for classification by SVM');

sv = SMOSVMModel.SupportVectors; %scatter plot of data points and points on support vectors
figure
gscatter(X(:,1),X(:,2),Y)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
title('SMO')
legend('Minor','Major','Support Vector')
hold off
% 
% CrossSVMModel = crossval(SMOSVMModel); %default 10-fold crossvalidation
% classLossSMO = kfoldLoss(CrossSVMModel); %missclassification after crossvalidation
% 
% yfit = predict(CrossSVMModel,newX); 
% cm = confusionchart(newY,yfit)
% title('CrossSVMModel')

SMOMdl = fitcsvm(X,Y,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus')) %optimizing model

NewSMOSVMModel = fitcsvm(X,Y,'KernelFunction','RBF', 'Solve', 'SMO',...
    'BoxConstraint',SMOMdl.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
    'KernelScale',SMOMdl.HyperparameterOptimizationResults.XAtMinObjective.KernelScale);

newLSMO = loss(NewSMOSVMModel, newX, newY); %misclassifcation after model has been completely optimized

yfit2 = predict(NewSMOSVMModel,newX); 
cm2 = confusionchart(newY,yfit2)
title('NewSMOSVMModel')

% [x2,y2,~,auc2] = perfcurve(newY,yfit2,1);
% plot(x2,y2)
% xlabel('False positive rate'); ylabel('True positive rate');
% title('ROC for classification by SVM');

%Now onto SofterMargin

SoftSVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF',...
    'Solver', 'L1QP'); %SoftMargin SVM uses L1QP

Lsoft = loss(SoftSVMModel,newX,newY); %raw missclassification 

yfit = predict(SoftSVMModel,newX); 
cm = confusionchart(newY,yfit)
title('SoftSVMModel')

sv = SoftSVMModel.SupportVectors; %scatter plot after using SoftMargin
figure
gscatter(X(:,1),X(:,2),Y)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
title('SoftMargin')
legend('Minor','Major','Support Vector')
hold off

% CrossSoftSVMModel = crossval(SoftSVMModel); %10-fold crossvalidation for SoftMargin
% classLossSoft = kfoldLoss(CrossSoftSVMModel); %missclassifcation after crossvalidation
% 
% yfit = predict(CrossSoftSVMModel,newX); 
% cm = confusionchart(newY,yfit)
% title('CrossSoftSVMModel')

SoftMdl = fitcsvm(X,Y,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus')) %optimizing model

NewSoftSVMModel = fitcsvm(X,Y,'KernelFunction','RBF','Solver', 'L1QP',...
    'BoxConstraint',SoftMdl.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
    'KernelScale',SoftMdl.HyperparameterOptimizationResults.XAtMinObjective.KernelScale);

newLsoft = loss(NewSoftSVMModel, newX, newY); %miscalssifciaiton after fully optimized SoftMargin

yfit = predict(NewSoftSVMModel,newX); 
cm = confusionchart(newY,yfit)
title('NewSoftSVMModel')

LSMO
% classLossSMO
newLSMO
Lsoft
% classLossSoft
newLsoft