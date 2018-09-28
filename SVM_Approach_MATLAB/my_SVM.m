function avg_acc = my_SVM(classes, main_names, main_classes, num_time_ft, num_f_ft, num_of_folds, package, data_main_path)

indices = randperm(length(main_names))';
num_per_fold = length(main_names) / num_of_folds;
fold_index = 1;
total_acc  = 0;
for i = 1 : num_per_fold : length(main_names)
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    fprintf('Working on fold number %f . \n',fold_index)
    main_classes_test = main_classes(indices(i : i + num_per_fold - 1));
    temp              = main_classes;
    temp(indices(i : i + num_per_fold - 1)) = [];
    main_classes_train   = temp;
    
    main_names_test = main_names(indices(i : i + num_per_fold - 1));
    temp            = main_names;
    temp(indices(i : i + num_per_fold - 1)) = [];
    main_names_train   = temp;
    
    [train_data, train_label] = getSavedFeatures(main_names_train, main_classes_train, package, num_time_ft + num_f_ft, data_main_path);
    [test_data,  test_label]  = getSavedFeatures(main_names_test,  main_classes_test,  package, num_time_ft + num_f_ft, data_main_path);
    fprintf('We have %f snippets for trianing, %f snippets for testing. \n',length(train_label), length(test_label));
    
    whole_data = [train_data;test_data];
    for j = 1 : num_time_ft * 2
        whole_data(:,j) = normc(whole_data(:,j));
    end
    for j = (num_time_ft * 2 + 1) : (num_time_ft + num_f_ft) * 2
        whole_data(:,j) = whole_data(:,j) / max(abs(whole_data(:,j)));
    end
    
    nor_train_data      = whole_data(1:length(train_data),:);
    nor_test_data       = whole_data(length(train_data)+1 : length(whole_data),:);
    
    disp('Start training')
    a_svm               = fitcsvm(nor_train_data, train_label);
    
    disp('Training completed, testing starts')
    [pred_label_cell,~] = predict(a_svm, nor_test_data);  
    
    pred_label          = strings(length(pred_label_cell),1);
    for j = 1 : length(pred_label)
        pred_label(j)   = pred_label_cell(j);
    end
    
    conf_mat            = confusionmat(test_label, pred_label);
    correctness         = diag(conf_mat);
    acc                 = 0;
    for j = 1 : length(classes)
        acc = acc + correctness(j) / sum(conf_mat(j,:));      
    end
    acc = acc / length(classes);
    total_acc = acc + total_acc;
    fprintf('This fold has %f accuracy. \n',acc)
    
    fold_index = fold_index + 1;
end
avg_acc = total_acc / num_of_folds;
end