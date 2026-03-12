%% MEDICAL ML MODEL + VISUALIZATION (FULL SCRIPT)

clc;
clear;
close all;

%% 1️⃣ LOAD DATASET
% Example medical-style dataset (replace with your CSV if needed)
% Format assumption: last column = diagnosis (0/1 or class labels)

% For demo, we create synthetic patient data
rng(1);
numPatients = 200;

Age = randi([20 80],numPatients,1);
BloodPressure = randi([90 180],numPatients,1);
Cholesterol = randi([150 300],numPatients,1);
HeartRate = randi([60 120],numPatients,1);

% Disease label (0 = Healthy, 1 = Disease)
Disease = double(0.03*Age + 0.02*BloodPressure + ...
                 0.015*Cholesterol + 0.02*HeartRate + randn(numPatients,1) > 8);

data = table(Age,BloodPressure,Cholesterol,HeartRate,Disease);

%% 2️⃣ FEATURE MATRIX & LABELS
X = data{:,1:4};
Y = categorical(data.Disease);

%% 3️⃣ TRAIN TEST SPLIT
cv = cvpartition(Y,'HoldOut',0.25);

Xtrain = X(training(cv),:);
Ytrain = Y(training(cv));

Xtest = X(test(cv),:);
Ytest = Y(test(cv));

%% 4️⃣ TRAIN MACHINE LEARNING MODEL (SVM)
model = fitcsvm(Xtrain,Ytrain, ...
    'KernelFunction','rbf', ...
    'Standardize',true);

%% 5️⃣ PREDICTION
[Ypred,score] = predict(model,Xtest);

%% 6️⃣ MODEL ACCURACY
accuracy = sum(Ypred == Ytest)/numel(Ytest);
fprintf('Model Accuracy: %.2f%%\n', accuracy*100);

%% 7️⃣ CONFUSION MATRIX VISUALIZATION
figure;
confusionchart(Ytest,Ypred);
title('Confusion Matrix - Disease Prediction');

%% 8️⃣ ROC CURVE
% Convert categorical to numeric for ROC
Ytest_num = double(Ytest) - 1;

[Xroc,Yroc,~,AUC] = perfcurve(Ytest_num,score(:,2),1);

figure;
plot(Xroc,Yroc,'LineWidth',2);
grid on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(AUC), ')']);

%% 9️⃣ FEATURE VISUALIZATION
figure;
gscatter(data.Age,data.Cholesterol,data.Disease,'rb','ox');
xlabel('Age');
ylabel('Cholesterol');
title('Feature Distribution');

%% 🔟 FEATURE IMPORTANCE (using tree model)
treeModel = fitctree(Xtrain,Ytrain);
imp = predictorImportance(treeModel);

figure;
bar(imp);
xlabel('Features');
ylabel('Importance');
title('Feature Importance');

xticklabels({'Age','BP','Chol','HR'});


newPatient = [55, 145, 240, 85];   % [Age, BloodPressure, Cholesterol, HeartRate]

% Predict using the trained SVM model
[newPred, newScore] = predict(model, newPatient);

% Display result
if newPred == categorical(1)
    fprintf('Prediction: HIGH RISK of cardiac disease (Probability = %.2f%%)\n', newScore(2)*100);
else
    fprintf('Prediction: LOW RISK of cardiac disease (Probability = %.2f%%)\n', (1-newScore(2))*100);
end
