# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:57:59 2015

@author: cai-mj
"""

import numpy as np
import os
import yaml
import caffe
import cv2
from skimage.feature import hog
from skimage import color

def get_deep_net():
    """ 
    get pretrained deep network model 
    """
    caffe_root = '/home/cai-mj/programs/caffe-master'
    MEAN_FILE = caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    MODEL_FILE = caffe_root + '/python/feature/imgnet_feature.prototxt'
    PRETRAINED = caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'        
    net = caffe.Classifier(MODEL_FILE, PRETRAINED, gpu=False)
    caffe.set_phase_test()
    net.set_mean('data', np.load(MEAN_FILE))
    net.set_raw_scale('data', 255)
    net.set_channel_swap('data', (2,1,0))
    return net


def extract_cnn_grasp(seqs, dstfile):
    """
    extract deep convolutional network features for hand grasps\n
    seqs: list of sequence names\n
    dstfile: file to which features are written
    """
    print('Prepare cnn feature to ' + dstfile + '\n')
    myfile = open(dstfile, 'w')
    myfile.write("seqname,filename,side,grasp,feature...\n")
    for seq in seqs:
        #collect all yaml file names
        command = "ls " + seq + "/*.yml > filename.txt"
        os.system(command)
        f_files = open("filename.txt", "r")
        files = f_files.readlines()
        f_files.close()    
        for j in range(len(files)):
            #read from each yaml file
            ymlfile = files[j][:len(files[j])-1]
            print(ymlfile)
            f_yml = open(ymlfile, "r")
            yml_dict = yaml.load(f_yml.read())
            f_yml.close()
            seqname = str(yml_dict["seqname"])
            filename = str(yml_dict["filename"])
            is_hLvisible = yml_dict["lefthand"]["visible"]
            hL_grasp = yml_dict["lefthand"]["grasp"]
            item_split = hL_grasp.split()
            hL_grasp = "-".join(item_split)
            hL_xmin = yml_dict["lefthand"]["bndbox"]["xmin"]
            hL_ymin = yml_dict["lefthand"]["bndbox"]["ymin"]
            hL_xmax = yml_dict["lefthand"]["bndbox"]["xmax"]
            hL_ymax = yml_dict["lefthand"]["bndbox"]["ymax"]
            is_hRvisible = yml_dict["righthand"]["visible"]
            hR_grasp = yml_dict["righthand"]["grasp"]
            item_split = hR_grasp.split()
            hR_grasp = "-".join(item_split)
            hR_xmin = yml_dict["righthand"]["bndbox"]["xmin"]
            hR_ymin = yml_dict["righthand"]["bndbox"]["ymin"]
            hR_xmax = yml_dict["righthand"]["bndbox"]["xmax"]
            hR_ymax = yml_dict["righthand"]["bndbox"]["ymax"]
            img = caffe.io.load_image(img_dir+"/"+seqname+"/"+filename+".jpg")
            if is_hLvisible == 1:
                imgroi = img[hL_ymin:hL_ymax+1, hL_xmin:hL_xmax+1]
                net.predict([imgroi])
                feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
                myfile.write(str(seqname)+","+str(filename)+",left,")
                myfile.write(hL_grasp+",")
                for value in feat:
                    myfile.write(str(value) + ',')
                myfile.write('\n')
            if is_hRvisible == 1:
                imgroi = img[hR_ymin:hR_ymax+1, hR_xmin:hR_xmax+1]
                net.predict([imgroi])
                feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
                myfile.write(str(seqname)+","+str(filename)+",right,")
                myfile.write(hR_grasp+",")
                for value in feat:
                    myfile.write(str(value) + ',')
                myfile.write('\n')
    #        break
    #    break
    myfile.close()  
    
def extract_hog_grasp(seqs, dstfile):
    """
    extract hog features for hand grasps\n
    seqs: list of sequence names\n
    dstfile: file to which features are written
    """
    print('Prepare cnn feature to ' + dstfile + '\n')
    myfile = open(dstfile, 'w')
    myfile.write("seqname,filename,side,grasp,feature...\n")
    for seq in seqs:
        #collect all yaml file names
        command = "ls " + seq + "/*.yml > filename.txt"
        os.system(command)
        f_files = open("filename.txt", "r")
        files = f_files.readlines()
        f_files.close()    
        for j in range(len(files)):
            #read from each yaml file
            ymlfile = files[j][:len(files[j])-1]
            print(ymlfile)
            f_yml = open(ymlfile, "r")
            yml_dict = yaml.load(f_yml.read())
            f_yml.close()
            seqname = str(yml_dict["seqname"])
            filename = str(yml_dict["filename"])
            is_hLvisible = yml_dict["lefthand"]["visible"]
            hL_grasp = yml_dict["lefthand"]["grasp"]
            item_split = hL_grasp.split()
            hL_grasp = "-".join(item_split)
            hL_xmin = yml_dict["lefthand"]["bndbox"]["xmin"]
            hL_ymin = yml_dict["lefthand"]["bndbox"]["ymin"]
            hL_xmax = yml_dict["lefthand"]["bndbox"]["xmax"]
            hL_ymax = yml_dict["lefthand"]["bndbox"]["ymax"]
            is_hRvisible = yml_dict["righthand"]["visible"]
            hR_grasp = yml_dict["righthand"]["grasp"]
            item_split = hR_grasp.split()
            hR_grasp = "-".join(item_split)
            hR_xmin = yml_dict["righthand"]["bndbox"]["xmin"]
            hR_ymin = yml_dict["righthand"]["bndbox"]["ymin"]
            hR_xmax = yml_dict["righthand"]["bndbox"]["xmax"]
            hR_ymax = yml_dict["righthand"]["bndbox"]["ymax"]
            img = cv2.imread(img_dir+"/"+seqname+"/"+filename+".jpg")
            if is_hLvisible == 1:
                imgroi = img[hL_ymin:hL_ymax+1, hL_xmin:hL_xmax+1]
                imgroi = color.rgb2gray(imgroi)
                imgroi = cv2.resize(imgroi, (80,80))
                feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                feat = feat_hog.tolist()
                myfile.write(str(seqname)+","+str(filename)+",left,")
                myfile.write(hL_grasp+",")
                for value in feat:
                    myfile.write(str(value) + ',')
                myfile.write('\n')
            if is_hRvisible == 1:
                imgroi = img[hR_ymin:hR_ymax+1, hR_xmin:hR_xmax+1]
                imgroi = color.rgb2gray(imgroi)
                imgroi = cv2.resize(imgroi, (80,80))
                feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                feat = feat_hog.tolist()
                myfile.write(str(seqname)+","+str(filename)+",right,")
                myfile.write(hR_grasp+",")
                for value in feat:
                    myfile.write(str(value) + ',')
                myfile.write('\n')
    #        break
    #    break
    myfile.close()  
    
    
#get pretrained network
LAYER = 'fc6wi'
INDEX = 4    
net = get_deep_net()
#dataset setting
img_dir = "/home/cai-mj/_GTA/img/GTEA"
seqs_train = []
seqs_train.append("006")
seqs_train.append("007")
seqs_train.append("008")
seqs_train.append("010")
seqs_train.append("012")
seqs_train.append("013")
seqs_train.append("014")
seqs_train.append("016")
seqs_train.append("017")
seqs_train.append("018")
seqs_train.append("021")
seqs_train.append("022")
seqs_test = []
seqs_test.append("002")
seqs_test.append("003")
seqs_test.append("005")
seqs_test.append("020")
trainfile = 'data/feature_grasp_train.csv'
testfile = 'data/feature_grasp_test.csv'
extract_cnn_grasp(seqs_train, trainfile)
extract_cnn_grasp(seqs_test, testfile)
