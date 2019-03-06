# ===============================================
# Import Packages and Functions
import random


# ===============================================
# Main Function
def splitData(name_class_combo, train_percent, validate_percent, test_percent):


    # ===============================================
    # Name Class Combo, format: [FileName, Type], example: ["AAAAA.wav", "Normal"]
    random.shuffle(name_class_combo)    


    # ===============================================
    # Compute the number of files for each set
    number_train    = round(len(name_class_combo) * train_percent    / 100)
    number_validate = round(len(name_class_combo) * validate_percent / 100)
    

    # ===============================================
    # Get the three sets 
    train_combo     = name_class_combo[0                              : number_train]
    validate_combo  = name_class_combo[number_train                   : number_train + number_validate]
    test_combo      = name_class_combo[number_train + number_validate : len(name_class_combo)]
    

    # ===============================================
    return train_combo, validate_combo, test_combo