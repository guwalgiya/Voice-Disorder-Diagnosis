from os import walk
def main(dataset_main_path, classes, slash):
    classes = ["Normal", "Pathol"]
    name_class_combo = []
    for a_class in classes:
        main_names = []
        for (dirpath, dirnames, filenames) in walk(dataset_main_path + slash + a_class):
            main_names.extend(filenames)
            break
        for i in range(len(main_names)):
            main_names[i] =  main_names[i][0 : len(main_names[i]) - 4]       
            name_class_combo.append([main_names[i], a_class])
    
    return name_class_combo
