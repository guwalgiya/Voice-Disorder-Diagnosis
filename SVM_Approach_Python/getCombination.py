# ===============================================
# Import Packages and Functions
from os import walk


# ===============================================
# main function
def getCombination(dataset_path, classes, slash):
    

    # ===============================================
    all_combo = []
    for a_class in classes:

    	
    	# ===============================================
        # main_names include all filenames with extension, for example: ["aaa.wav", "bbb.wav"]
        main_names = []
        for (dirpath, dirnames, filenames) in walk(dataset_path + slash + a_class):
            main_names.extend(filenames)
            break


        # ===============================================
        # remove .wav extension, and creat combos, for example, [["aaa", "Normal"], ["bbb", "Pathol"]]
        for i in range(len(main_names)):
            main_names[i] =  main_names[i][0 : len(main_names[i]) - 4]       
            all_combo.append([main_names[i], a_class])
    

    # ===============================================    
    return all_combo
