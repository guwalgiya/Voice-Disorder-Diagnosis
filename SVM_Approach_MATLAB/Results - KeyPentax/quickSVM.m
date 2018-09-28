
[features,txt,~] = xlsread("KeyPentax_Binary_Feature.csv");
[m, n]           = size(txt);
num_of_data      = m - 1;
label            = txt(2 : m, n);
indices          = randperm(length(label))';
num_of_folds = 5;
%num_per_fold = length(main_names) / num_of_folds;
num_per_fold = 142;
fold_index = 1;
total_acc  = 0;
num_of_classes    = 2;
for i = 1 : num_per_fold : num_of_data
%disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
%fprintf('Working on fold number %f . \n',fold_index);

end_point = min(i + num_per_fold - 1, num_of_data);

test_label = label(indices(i : end_point));
temp              = label;
temp(indices(i : end_point)) = [];
train_label       = temp;

test_data    = features(indices(i : end_point),:);
temp         = features;
temp(indices(i : end_point),:) = [];
train_data   = temp;


%disp('Start training')
a_svm               = fitcsvm(train_data, train_label);

%disp('Training completed, testing starts')
[pred_label,~] = predict(a_svm, test_data);  

conf_mat            = confusionmat(test_label, pred_label);
correctness         = diag(conf_mat);
acc                 = 0;
for j = 1 : num_of_classes
    acc = acc + correctness(j) / sum(conf_mat(j,:));   
end
acc = acc / num_of_classes;
total_acc = acc + total_acc;
%fprintf('This fold has %f accuracy. \n',acc)

fold_index = fold_index + 1;
end
avg_acc = total_acc / num_of_folds;
disp(avg_acc)