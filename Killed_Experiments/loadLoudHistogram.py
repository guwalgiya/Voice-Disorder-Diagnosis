import numpy as np
from os import walk
def main(x_combo, dType, classes, package, dataset_main_path):
    print('Now, loading ', dType, ' data')
    if x_combo == []:
        return [], [], {} 
    
    
    snippet_length, snippet_hop, bin_size = package
    
    x_distribution = {}
    for a_class in classes:
        x_distribution[a_class] = [0,0] #[0 originial files, 0 snippets]
    
    
    x_label, whole_path = [], []    
    for a_combo in x_combo:
        sub_folder       = a_combo[1] + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
        histogram_folder = a_combo[0] + "_LoudHistogram_bin" + str(bin_size)        
        histogram_path   = dataset_main_path + "/" + sub_folder + "/" + histogram_folder
        
        histogram_names = []
        for (dirpath, dirnames, filenames) in walk(histogram_path):
            histogram_names.extend(filenames)
            break
        
        for i in range(len(histogram_names)):
            whole_path.append(histogram_path + "/" + histogram_names[i])
        
        #a_combo_label                 = [0] * len(classes)
        #class_index                   = classes.index(a_combo[1])
        #a_combo_label[class_index]    = 1            
        #x_label                       = x_label + [a_combo_label] * len(histogram_names)
        
        x_label                        = x_label + [a_combo[1]] * len(histogram_names)

        x_distribution[a_combo[1]][0] = x_distribution[a_combo[1]][0] + 1
        x_distribution[a_combo[1]][1] = x_distribution[a_combo[1]][1] + len(histogram_names)
                       
    x_loaded = np.zeros((len(x_label), bin_size))
    
    for i in range(len(whole_path)):
        x_loaded[i] = np.loadtxt(whole_path[i])
        #if i % 10000 == 0 and i!= 0: 
    
    print(len(whole_path),'is done' )
    
    #x_loaded = x_loaded.reshape((len(x_loaded), np.prod(x_loaded.shape[1:])))    
    print(dType, ' data is loaded')
    print('******************************************************************')    
    return x_loaded, x_label, x_distribution# -*- coding: utf-8 -*-

