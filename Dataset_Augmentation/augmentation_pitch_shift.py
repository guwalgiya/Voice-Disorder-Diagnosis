import subprocess
import math
from os import walk
import numpy as np

def pitch_shifting(vocal_type, pitch_shift_tool, num_step):
    original_path = "C:\\Master Degree\\7100 - Master Project\\Dataset - Spanish" + "\\" + vocal_type;

    file_list = []
    for (dirpath, dirnames, filenames) in walk(original_path):
        file_list.extend(filenames)
        break

    output_path_preset   = "C:\\Master Degree\\7100 - Master Project\\Dataset - Spanish" + "\\" + vocal_type + "_Pitch_Shifted"
    for filename in file_list:
        input_path = original_path + "\\" + filename

        for step in np.linspace(0, num_step, num_step / 0.5 + 1):
            if step == 0: 
                output_path = output_path_preset + "\\" + filename[0: len(filename) - 4] + "_" + "N" + str(step) + ".wav"
                subprocess.Popen([pitch_shift_tool, "-i", input_path, "-o", output_path, "-s", "1", "-p", "1"]).wait()
                print(filename, step, 'done')
                
            else:
                output_path = output_path_preset + "\\" + filename[0: len(filename) - 4] + "_" + "U" + str(step) + ".wav"
                ratio = math.pow(2, step * 100 / 1200)
                subprocess.Popen([pitch_shift_tool, "-i", input_path, "-o", output_path, "-s", "1", "-p", str(ratio)]).wait()
                print(filename, step, 'done')
                
                output_path = output_path_preset + "\\" + filename[0: len(filename) - 4] + "_" + "D" + str(step) + ".wav"
                ratio = math.pow(2, -1 * step * 100 / 1200)
                subprocess.Popen([pitch_shift_tool, "-i", input_path, "-o", output_path, "-s", "1", "-p", str(ratio)]).wait()
                print(filename, -1 * step, 'done')


# User Input
pitch_shift_tool = "C:\\Master Degree\\7100 - Master Project\\elastiqueProCl.exe";
fs = 25000
num_step = 4
vocal_types = ["Normal", "Pathol"]
for vocal_type in vocal_types:
    pitch_shifting(vocal_type, pitch_shift_tool, num_step)       