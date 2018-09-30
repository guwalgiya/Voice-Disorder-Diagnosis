import random
def main(name_class_combo, train_percent, validate_percent, test_percent):
    #print('Split Data')
    random.shuffle(name_class_combo)    
    number_train    = round(len(name_class_combo) * train_percent    / 100)
    number_validate = round(len(name_class_combo) * validate_percent / 100)
    #number_test     = len(name_class_combo) - number_train - number_validate
    train_combo     = name_class_combo[0 : number_train]
    validate_combo  = name_class_combo[number_train : number_train + number_validate]
    test_combo      = name_class_combo[number_train + number_validate : len(name_class_combo)]
    
    return train_combo, validate_combo, test_combo
