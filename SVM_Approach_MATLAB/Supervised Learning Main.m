%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dataset set up setion
% Don't forget to do "add to path" for the folder "Feature Extraction" !!!
working_path   = "/home/hguan/Vocal-Disorder-Diagnosis/Supervised learning Approach"
data_main_path = "/home/hguan/7100-Master-Project/Dataset-KeyPentax"
classes        = ["Normal","Pathol"];
snippet_length = 500;
snippet_hop    = 200;
block_size     = 512;
hop_size       = 256;
package        = [snippet_length, snippet_hop, block_size, hop_size];
num_time_ft    = 4;
num_f_ft       = 31;
num_features   = num_time_ft + num_f_ft;
num_of_folds   = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Count the number of original Files; get their names and their types
disp("Step 1")
[main_names, main_classes] = getGroundTruth(data_main_path, classes); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Extract and Save features for all the snippet
disp("Step 2")
getAudioFeatures(num_features,data_main_path, working_path, main_names, main_classes, package);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Do 10-cross validation SVM
disp("Step 3")
final_acc_500  = my_SVM(classes, main_names, main_classes, num_time_ft, num_f_ft, num_of_folds, package, data_main_path);

package        = [600, snippet_hop, block_size, hop_size];
final_acc_600  = my_SVM(classes, main_names, main_classes, num_time_ft, num_f_ft, num_of_folds, package, data_main_path);

package        = [700, snippet_hop, block_size, hop_size];
final_acc_700  = my_SVM(classes, main_names, main_classes, num_time_ft, num_f_ft, num_of_folds, package, data_main_path);


disp(final_acc_500)
disp(final_acc_600)
disp(final_acc_700)




