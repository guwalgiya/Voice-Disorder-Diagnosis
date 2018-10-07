import getCombination
import getMelSpectrogram
import getLoudnessHistogram
import getMelSpectrogram_for_unaugmented
import getMFCCs
import getMFCCs_for_unaugmented
def main():
    task    = input("Choose the task \nMel Spectrogram = 1 \nMFCCs = 2 \n")
    options = {1 : melSpectrogram,
               2 : MFCCs
               } 
    options[int(task)]()
    #melSpectrogram()
    print("Done")

def MFCCs():
    slash, dataset_main_path                          = os_choose()
    para_values                                       = ask_parameters(["blocksize", "hopsize"])
    classes, fs, snippet_length, snippet_hop          = ask_defaults()
    block_size                                        = para_values[0]
    hop_size                                          = para_values[1]
    name_class_combo                                  = getCombination.main(dataset_main_path, classes, slash)
    #getMFCCs.main(dataset_main_path, name_class_combo, fs, snippet_length, snippet_hop, block_size, hop_size)
    #getMFCCs_for_testing.main(dataset_main_path, name_class_combo, fs, snippet_length, snippet_hop, block_size, hop_size)

def melSpectrogram():
    slash, dataset_main_path                          = os_choose()
    para_values                                       = ask_parameters(["blocksize", "hopsize", "melength"])
    classes, fs, snippet_length, snippet_hop          = ask_defaults()
    block_size                                        = para_values[0]
    hop_size                                          = para_values[1]
    mel_length                                        = para_values[2]
    name_class_combo                                  = getCombination.main(dataset_main_path, classes, slash)

    getMFCCs.main(dataset_main_path, name_class_combo, fs, snippet_length, snippet_hop, block_size, hop_size)
    getMFCCs_for_unaugmented.main(dataset_main_path, name_class_combo, fs, snippet_length, snippet_hop, block_size, hop_size)
    getMelSpectrogram.main(dataset_main_path, name_class_combo, fs, snippet_length, snippet_hop, block_size, hop_size, mel_length)
    getMelSpectrogram_for_unaugmented.main(dataset_main_path, name_class_combo, fs, snippet_length, snippet_hop, block_size, hop_size, mel_length)


def loudnessHistogram():
    slash, dataset_main_path                          = os_choose()
    para_values                                       = ask_parameters(['bin_size'])
    classes, fs, snippet_length, snippet_hop          = ask_defaults()
    bin_size                                          = para_values[0]
    name_class_combo                                  = getCombination.main(dataset_main_path, classes, slash)

    getLoudnessHistogram.main(dataset_main_path, name_class_combo, fs, snippet_length, snippet_hop, bin_size, slash)
    
    
def os_choose():
    correct_input = False
    while(not correct_input):
        os = input("Choose Operating System: Linux = 1, Windows = 2 ")
        if int(os)            == 1:
            slash              = "/"
            dataset_main_path  = "/home/hguan/7100-Master-Project/Dataset-Spanish"
            correct_input      = True
        elif int(os)          == 2:
            slash              = "\\"
            dataset_main_path  = "C:\\Music Technology Master\\7100 - Master Project\\Dataset - Spanish"
            correct_input      = True
        else:
            pass
    return slash,dataset_main_path

def ask_parameters(para_names):
    para_values = []
    for parameter in para_names:
        print("\n")
        print(parameter)
        value = input("What is the value for it ?? ")
        para_values.append(int(value))
        print("\n")
    return para_values

def ask_defaults():
    classes = ["Normal", "Pathol"]
    fs               = 16000
    snippet_length   = 500
    snippet_hop      = 100

    return classes, fs, snippet_length, snippet_hop
    
main()    
