#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:40:05 2017

@author: harshavardhan
"""

from PIL import Image
import glob
import csv
#import numpy as np


def extract_features(img):
    data = img.getdata()
    rw = len(data)
    col = len(data[0])
    features = []
    for j in range(col):
        bins = [0]*32
        for i in range(rw):
            bins[data[i][j]//8] +=1
        features.extend(bins)
    
    return features


def write_to_file(filename, features):
    target_dir = '../../Dataset/'
    with open(target_dir+filename,'w') as f:
        cwriter = csv.writer(f)
        cwriter.writerows(features)

def main():
    classes = ['mountain', 'forest','insidecity','coast']
    datatype = ['Test', 'Train']
    label = {}
    label['mountain'] = [1,0,0,0]
    label['forest'] = [0,1,0,0]
    label['insidecity'] = [0,0,1,0]
    label['coast'] = [0,0,0,1]


    for dtype in datatype:
        filename = 'DS2full-'+dtype.lower()+'.csv'
        all_features = []
        for clas in classes:
            for input_file in glob.glob('../../Dataset/Data_LR(DS2)/data_students/'+clas+'/'+dtype+'/*.jpg'):
                img = Image.open(input_file)
                datafeatures = extract_features(img)
                datafeatures.extend(label[clas])
                all_features.append(datafeatures)
        #np.random.shuffle(all_features)
        write_to_file(filename,all_features)

if __name__ == '__main__':
    main()