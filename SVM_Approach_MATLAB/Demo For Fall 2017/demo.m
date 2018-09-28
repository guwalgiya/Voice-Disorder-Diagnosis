clc
clear
disp('Vocal Disorder Diagnosis')
fs = 25000;
count_down = 3;
fprintf('Say Ah after %d seconds count down',count_down)
disp('                         ')
pause(2)
recObject = audiorecorder(fs,8,1);
for i = 1 : count_down
    fprintf('%d !!!!!!!!!!!!!!!!!!!!!!!!!!!',count_down + 1 - i)
    disp('                      ')
    pause(1)
end


disp('Keep saying "ahhhhhhhhhhhhhhhhhhh" for 3 seconds')
recordblocking(recObject,3)
x = getaudiodata(recObject);
x = x(length(x)/2 : end);
disp('Recording is done')
disp('Doing Feature Extraction')
pause(1)
rawFeatures = extractFeatures(x, 1024, 512, fs);
[numFeatures, ~] = size(rawFeatures);
aggregatedFeatureVec = [];
for i = 1 : numFeatures
    feature = rawFeatures(i,:);
    aggregatedFeatureVec = [aggregatedFeatureVec,mean(feature),std(feature)];
end


load('spanish_miu.mat')
load('spanish_sd.mat')
for i = 1 : length(aggregatedFeatureVec)
    aggregatedFeatureVec(i) = (aggregatedFeatureVec(i) - spanish_miu(i)) / spanish_sd(i);
end

disp('%%%%%%%%%%%%%%%%%%%%%%')
disp('Training')
load('spanish_training.mat')
load('ground_truth_level_0.mat')
load('ground_truth_level_1.mat')
Model_0 = fitctree(spanish_training,ground_truth_level_0);
predict_0 = predict(Model_0,aggregatedFeatureVec);
pause(1)
disp('Binary Classification is Ready!')

Model_1 = fitctree(spanish_training,ground_truth_level_1);
predict_1 = predict(Model_1,aggregatedFeatureVec);
pause(1)
disp('Multiple Classification is Ready!')


disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%')
pause(2)
switch predict_0
    case 0
        disp('The first model says you are healthy')
    case 1
        disp('The first model says you have vocal disorder')
end


disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%')
pause(2)
switch predict_1
    case 0
        disp('The second model says you are healthy')   
    case 1
        disp('The second model says you have vocal disorder')
        disp('Type - Organic Pathology')
    case 2
        disp('The second model says you have vocal disorder')
        disp('Type - Minimal Associated Injuries')
    case 3
        disp('The second model says you have vocal disorder')
        disp('Type - Functional')
end

