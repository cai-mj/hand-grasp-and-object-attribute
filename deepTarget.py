# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:27:10 2015

@author: cai-mj
"""

import numpy as np
import os
import yaml
import math
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
    
    
def extract_cnn_target(seqs, dstfile):
    """
    extract deep convolutional network features for targe object location\n
    seqs: list of sequence names\n
    dstfile: file to which features are written
    """
    print('Prepare cnn feature to ' + dstfile + '\n')
    dim = 0
    myfile = open(dstfile, 'w')
    myfile.write("seqname,filename,side,relativeX,relativeY,relativeS,feature...\n")
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
            hL_xmin = yml_dict["lefthand"]["bndbox"]["xmin"]
            hL_ymin = yml_dict["lefthand"]["bndbox"]["ymin"]
            hL_xmax = yml_dict["lefthand"]["bndbox"]["xmax"]
            hL_ymax = yml_dict["lefthand"]["bndbox"]["ymax"]
            is_oLvisible = yml_dict["leftobject"]["visible"]
            oL_xmin = yml_dict["leftobject"]["bndbox"]["xmin"]
            oL_ymin = yml_dict["leftobject"]["bndbox"]["ymin"]
            oL_xmax = yml_dict["leftobject"]["bndbox"]["xmax"]
            oL_ymax = yml_dict["leftobject"]["bndbox"]["ymax"]
            is_hRvisible = yml_dict["righthand"]["visible"]
            hR_xmin = yml_dict["righthand"]["bndbox"]["xmin"]
            hR_ymin = yml_dict["righthand"]["bndbox"]["ymin"]
            hR_xmax = yml_dict["righthand"]["bndbox"]["xmax"]
            hR_ymax = yml_dict["righthand"]["bndbox"]["ymax"]
            is_oRvisible = yml_dict["rightobject"]["visible"]
            oR_xmin = yml_dict["rightobject"]["bndbox"]["xmin"]
            oR_ymin = yml_dict["rightobject"]["bndbox"]["ymin"]
            oR_xmax = yml_dict["rightobject"]["bndbox"]["xmax"]
            oR_ymax = yml_dict["rightobject"]["bndbox"]["ymax"]
            img = caffe.io.load_image(img_dir+"/"+seqname+"/"+filename+".jpg")
            if is_hLvisible == 1 and is_oLvisible == 1:
                imgroi = img[hL_ymin:hL_ymax+1, hL_xmin:hL_xmax+1]
                net.predict([imgroi])
                feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
                relativeX = ((oL_xmin+oL_xmax)/2-(hL_xmin+hL_xmax)/2)/float(hL_xmax-hL_xmin+1)
                relativeY = ((oL_ymin+oL_ymax)/2-(hL_ymin+hL_ymax)/2)/float(hL_ymax-hL_ymin+1)
                relativeS = math.sqrt((oL_xmax-oL_xmin+1)*(oL_ymax-oL_ymin+1))/math.sqrt((hL_xmax-hL_xmin+1)*(hL_ymax-hL_ymin+1))
                myfile.write(str(seqname)+","+str(filename)+",left,")
                myfile.write(str(relativeX)+",")
                myfile.write(str(relativeY)+",")
                myfile.write(str(relativeS)+",")
                for value in feat:
                    myfile.write(str(value) + ',')
                myfile.write('\n')
                dim = len(feat)
            if is_hRvisible == 1 and is_oRvisible == 1:
                imgroi = img[hR_ymin:hR_ymax+1, hR_xmin:hR_xmax+1]
                net.predict([imgroi])
                feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
                relativeX = ((oR_xmin+oR_xmax)/2-(hR_xmin+hR_xmax)/2)/float(hR_xmax-hR_xmin+1)
                relativeY = ((oR_ymin+oR_ymax)/2-(hR_ymin+hR_ymax)/2)/float(hR_ymax-hR_ymin+1)
                relativeS = math.sqrt((oR_xmax-oR_xmin+1)*(oR_ymax-oR_ymin+1))/math.sqrt((hR_xmax-hR_xmin+1)*(hR_ymax-hR_ymin+1))
                myfile.write(str(seqname)+","+str(filename)+",right,")
                myfile.write(str(relativeX)+",")
                myfile.write(str(relativeY)+",")
                myfile.write(str(relativeS)+",")
                for value in feat:
                    myfile.write(str(value) + ',')
                myfile.write('\n')
                dim = len(feat)
    #        break
    #    break
    myfile.close()  
    print "feature dimension: ", dim
    
def extract_hog_target(seqs, dstfile):
    """
    extract hog features for targe object location\n
    seqs: list of sequence names\n
    dstfile: file to which features are written
    """
    print('Prepare cnn feature to ' + dstfile + '\n')
    dim = 0
    myfile = open(dstfile, 'w')
    myfile.write("seqname,filename,side,relativeX,relativeY,relativeS,feature...\n")
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
            hL_xmin = yml_dict["lefthand"]["bndbox"]["xmin"]
            hL_ymin = yml_dict["lefthand"]["bndbox"]["ymin"]
            hL_xmax = yml_dict["lefthand"]["bndbox"]["xmax"]
            hL_ymax = yml_dict["lefthand"]["bndbox"]["ymax"]
            is_oLvisible = yml_dict["leftobject"]["visible"]
            oL_xmin = yml_dict["leftobject"]["bndbox"]["xmin"]
            oL_ymin = yml_dict["leftobject"]["bndbox"]["ymin"]
            oL_xmax = yml_dict["leftobject"]["bndbox"]["xmax"]
            oL_ymax = yml_dict["leftobject"]["bndbox"]["ymax"]
            is_hRvisible = yml_dict["righthand"]["visible"]
            hR_xmin = yml_dict["righthand"]["bndbox"]["xmin"]
            hR_ymin = yml_dict["righthand"]["bndbox"]["ymin"]
            hR_xmax = yml_dict["righthand"]["bndbox"]["xmax"]
            hR_ymax = yml_dict["righthand"]["bndbox"]["ymax"]
            is_oRvisible = yml_dict["rightobject"]["visible"]
            oR_xmin = yml_dict["rightobject"]["bndbox"]["xmin"]
            oR_ymin = yml_dict["rightobject"]["bndbox"]["ymin"]
            oR_xmax = yml_dict["rightobject"]["bndbox"]["xmax"]
            oR_ymax = yml_dict["rightobject"]["bndbox"]["ymax"]
            img = cv2.imread(img_dir+"/"+seqname+"/"+filename+".jpg")
            if is_hLvisible == 1 and is_oLvisible == 1:
                imgroi = img[hL_ymin:hL_ymax+1, hL_xmin:hL_xmax+1]
                imgroi = color.rgb2gray(imgroi)
                imgroi = cv2.resize(imgroi, (80,80))
                feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                feat = feat_hog.tolist()
                relativeX = ((oL_xmin+oL_xmax)/2-(hL_xmin+hL_xmax)/2)/float(hL_xmax-hL_xmin+1)
                relativeY = ((oL_ymin+oL_ymax)/2-(hL_ymin+hL_ymax)/2)/float(hL_ymax-hL_ymin+1)
                relativeS = math.sqrt((oL_xmax-oL_xmin+1)*(oL_ymax-oL_ymin+1))/math.sqrt((hL_xmax-hL_xmin+1)*(hL_ymax-hL_ymin+1))
                myfile.write(str(seqname)+","+str(filename)+",left,")
                myfile.write(str(relativeX)+",")
                myfile.write(str(relativeY)+",")
                myfile.write(str(relativeS)+",")
                for value in feat:
                    myfile.write(str(value) + ',')
                myfile.write('\n')
                dim = len(feat)
            if is_hRvisible == 1 and is_oRvisible == 1:
                imgroi = img[hR_ymin:hR_ymax+1, hR_xmin:hR_xmax+1]
                imgroi = color.rgb2gray(imgroi)
                imgroi = cv2.resize(imgroi, (80,80))
                feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                feat = feat_hog.tolist()
                relativeX = ((oR_xmin+oR_xmax)/2-(hR_xmin+hR_xmax)/2)/float(hR_xmax-hR_xmin+1)
                relativeY = ((oR_ymin+oR_ymax)/2-(hR_ymin+hR_ymax)/2)/float(hR_ymax-hR_ymin+1)
                relativeS = math.sqrt((oR_xmax-oR_xmin+1)*(oR_ymax-oR_ymin+1))/math.sqrt((hR_xmax-hR_xmin+1)*(hR_ymax-hR_ymin+1))
                myfile.write(str(seqname)+","+str(filename)+",right,")
                myfile.write(str(relativeX)+",")
                myfile.write(str(relativeY)+",")
                myfile.write(str(relativeS)+",")
                for value in feat:
                    myfile.write(str(value) + ',')
                myfile.write('\n')
                dim = len(feat)
    #        break
    #    break
    myfile.close()  
    print "feature dimension: ", dim
    
    
#get pretrained network
LAYER = 'pool5'
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
trainfile = 'data/feature_target_train.csv'
testfile = 'data/feature_target_test.csv'
extract_hog_target(seqs_train, trainfile)
extract_hog_target(seqs_test, testfile)