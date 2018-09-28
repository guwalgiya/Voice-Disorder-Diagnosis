# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:07:36 2017

@author: GuwalgiyaGuan
"""

import csv

mylist2 = []
f = open('Pathol_Filename_Type_Original.csv', 'r')
mylist = csv.reader(f)
for pair in mylist:
    mylist2.append([pair[0],int(pair[1][0])])
f.close()



file = open("Filename_Type_Level1.csv", 'w',newline = "")
csv_file = csv.writer(file)
for item2 in mylist2:
    csv_file.writerow(item2)    
file.close()