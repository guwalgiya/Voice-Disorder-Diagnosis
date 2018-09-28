# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:07:36 2017

@author: GuwalgiyaGuan
"""

import csv

d = {}
f = open('Filename_Type_MatchingTable.csv', 'r')
mylist = csv.reader(f)
for pair in mylist:
    filename = pair[0][0:-3]
    filename = "'" + filename + "wav'"
    if pair[1] == ' ':
        d[filename] = 'Undefined'
    else:    
        d[filename] = pair[1]
f.close()


f2 = open('RawDataSet.csv','r')
mylist2 = csv.reader(f2)
addingList = ['']

for item in mylist2:
    for name in item:
        try:
            addingList.append(d[name])
        except:
            pass
    break
    
    
print(addingList)
file = open("Classification_Multiple.csv", 'w',newline = "")
csv_file = csv.writer(file)
csv_file.writerow(addingList)
for item2 in mylist2:
    csv_file.writerow(item2)    
file.close()
f2.close()