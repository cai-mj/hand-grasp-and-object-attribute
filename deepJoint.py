# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:44:38 2015

@author: cai-mj
"""

import numpy as np
import os
import yaml
import caffe
from common import transfer_grasp_label
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
    
def extract_cnn_grasp_attribute(seqs):
    """
    extract deep convolutional network features for hand grasps\n
    seqs: list of sequence names\n
    dstfile: file to which features are written
    """
    g_a_left = {}
    g_a_right = {}
    for seq in seqs:
        #collect all yaml file names
        print('Prepare cnn feature from ' + seq + '...')
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
            hL_grasp = transfer_grasp_label(hL_grasp)
            hL_xmin = yml_dict["lefthand"]["bndbox"]["xmin"]
            hL_ymin = yml_dict["lefthand"]["bndbox"]["ymin"]
            hL_xmax = yml_dict["lefthand"]["bndbox"]["xmax"]
            hL_ymax = yml_dict["lefthand"]["bndbox"]["ymax"]
            is_hRvisible = yml_dict["righthand"]["visible"]
            hR_grasp = yml_dict["righthand"]["grasp"]
            item_split = hR_grasp.split()
            hR_grasp = "-".join(item_split)
            hR_grasp = transfer_grasp_label(hR_grasp)
            hR_xmin = yml_dict["righthand"]["bndbox"]["xmin"]
            hR_ymin = yml_dict["righthand"]["bndbox"]["ymin"]
            hR_xmax = yml_dict["righthand"]["bndbox"]["xmax"]
            hR_ymax = yml_dict["righthand"]["bndbox"]["ymax"]
            is_oLvisible = yml_dict["leftobject"]["visible"]
            oL_xmin = yml_dict["leftobject"]["bndbox"]["xmin"]
            oL_ymin = yml_dict["leftobject"]["bndbox"]["ymin"]
            oL_xmax = yml_dict["leftobject"]["bndbox"]["xmax"]
            oL_ymax = yml_dict["leftobject"]["bndbox"]["ymax"]
            is_oLprismatic = yml_dict["leftobject"]["attribute"]["prismatic"]
            is_oLsphere = yml_dict["leftobject"]["attribute"]["sphere"]
            is_oLflat = yml_dict["leftobject"]["attribute"]["flat"]
            is_oLrigid = yml_dict["leftobject"]["attribute"]["rigid"]
            is_oRvisible = yml_dict["rightobject"]["visible"]
            oR_xmin = yml_dict["rightobject"]["bndbox"]["xmin"]
            oR_ymin = yml_dict["rightobject"]["bndbox"]["ymin"]
            oR_xmax = yml_dict["rightobject"]["bndbox"]["xmax"]
            oR_ymax = yml_dict["rightobject"]["bndbox"]["ymax"]
            is_oRprismatic = yml_dict["rightobject"]["attribute"]["prismatic"]
            is_oRsphere = yml_dict["rightobject"]["attribute"]["sphere"]
            is_oRflat = yml_dict["rightobject"]["attribute"]["flat"]
            is_oRrigid = yml_dict["rightobject"]["attribute"]["rigid"]
            img = caffe.io.load_image(img_dir+"/"+seqname+"/"+filename+".jpg")
            if is_hLvisible == 1 and is_oLvisible == 1:
                value_oneframe = []
                value_oneframe.append(str(seqname)+","+str(filename)+",left")
                value_oneframe.append(str(is_oLprismatic)+","+str(is_oLsphere)+","+str(is_oLflat)+","+str(is_oLrigid))
                imgroi = img[hL_ymin:hL_ymax+1, hL_xmin:hL_xmax+1]
                net.predict([imgroi])
                feat = net.blobs[LAYER].data[INDEX].flatten().tolist()                
                feat_str = ""
                for value in feat:
                    feat_str = feat_str + str(value) + ','
                value_oneframe.append(feat_str)
                imgroi = img[oL_ymin:oL_ymax+1, oL_xmin:oL_xmax+1]
                net.predict([imgroi])
                feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
                feat_str = ""
                for value in feat:
                    feat_str = feat_str + str(value) + ','
                value_oneframe.append(feat_str)
                if g_a_left.has_key(hL_grasp):
                    g_a_left[hL_grasp].append(value_oneframe)
                else:
                    g_a_left[hL_grasp] = []
                    g_a_left[hL_grasp].append(value_oneframe)
            if is_hRvisible == 1 and is_oRvisible:
                value_oneframe = []
                value_oneframe.append(str(seqname)+","+str(filename)+",right")
                value_oneframe.append(str(is_oRprismatic)+","+str(is_oRsphere)+","+str(is_oRflat)+","+str(is_oRrigid))
                imgroi = img[hR_ymin:hR_ymax+1, hR_xmin:hR_xmax+1]
                net.predict([imgroi])
                feat = net.blobs[LAYER].data[INDEX].flatten().tolist()                
                feat_str = ""
                for value in feat:
                    feat_str = feat_str + str(value) + ','
                value_oneframe.append(feat_str)
                imgroi = img[oR_ymin:oR_ymax+1, oR_xmin:oR_xmax+1]
                net.predict([imgroi])
                feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
                feat_str = ""
                for value in feat:
                    feat_str = feat_str + str(value) + ','
                value_oneframe.append(feat_str)
                if g_a_right.has_key(hR_grasp):
                    g_a_right[hR_grasp].append(value_oneframe)
                else:
                    g_a_right[hR_grasp] = []
                    g_a_right[hR_grasp].append(value_oneframe)
    #        break
    #    break 
    return g_a_left, g_a_right
    
def extract_hog_grasp_attribute(seqs):
    """
    extract hog features for hand grasps\n
    seqs: list of sequence names\n
    dstfile: file to which features are written
    """
    g_a_left = {}
    g_a_right = {}
    for seq in seqs:
        #collect all yaml file names
        print('Prepare cnn feature from ' + seq + '...')
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
            hL_grasp = transfer_grasp_label(hL_grasp)
            hL_xmin = yml_dict["lefthand"]["bndbox"]["xmin"]
            hL_ymin = yml_dict["lefthand"]["bndbox"]["ymin"]
            hL_xmax = yml_dict["lefthand"]["bndbox"]["xmax"]
            hL_ymax = yml_dict["lefthand"]["bndbox"]["ymax"]
            is_hRvisible = yml_dict["righthand"]["visible"]
            hR_grasp = yml_dict["righthand"]["grasp"]
            item_split = hR_grasp.split()
            hR_grasp = "-".join(item_split)
            hR_grasp = transfer_grasp_label(hR_grasp)
            hR_xmin = yml_dict["righthand"]["bndbox"]["xmin"]
            hR_ymin = yml_dict["righthand"]["bndbox"]["ymin"]
            hR_xmax = yml_dict["righthand"]["bndbox"]["xmax"]
            hR_ymax = yml_dict["righthand"]["bndbox"]["ymax"]
            is_oLvisible = yml_dict["leftobject"]["visible"]
            oL_xmin = yml_dict["leftobject"]["bndbox"]["xmin"]
            oL_ymin = yml_dict["leftobject"]["bndbox"]["ymin"]
            oL_xmax = yml_dict["leftobject"]["bndbox"]["xmax"]
            oL_ymax = yml_dict["leftobject"]["bndbox"]["ymax"]
            is_oLprismatic = yml_dict["leftobject"]["attribute"]["prismatic"]
            is_oLsphere = yml_dict["leftobject"]["attribute"]["sphere"]
            is_oLflat = yml_dict["leftobject"]["attribute"]["flat"]
            is_oLrigid = yml_dict["leftobject"]["attribute"]["rigid"]
            is_oRvisible = yml_dict["rightobject"]["visible"]
            oR_xmin = yml_dict["rightobject"]["bndbox"]["xmin"]
            oR_ymin = yml_dict["rightobject"]["bndbox"]["ymin"]
            oR_xmax = yml_dict["rightobject"]["bndbox"]["xmax"]
            oR_ymax = yml_dict["rightobject"]["bndbox"]["ymax"]
            is_oRprismatic = yml_dict["rightobject"]["attribute"]["prismatic"]
            is_oRsphere = yml_dict["rightobject"]["attribute"]["sphere"]
            is_oRflat = yml_dict["rightobject"]["attribute"]["flat"]
            is_oRrigid = yml_dict["rightobject"]["attribute"]["rigid"]
            img = cv2.imread(img_dir+"/"+seqname+"/"+filename+".jpg")
            if is_hLvisible == 1 and is_oLvisible == 1:
                value_oneframe = []
                value_oneframe.append(str(seqname)+","+str(filename)+",left")
                value_oneframe.append(str(is_oLprismatic)+","+str(is_oLsphere)+","+str(is_oLflat)+","+str(is_oLrigid))
                imgroi = img[hL_ymin:hL_ymax+1, hL_xmin:hL_xmax+1]
                imgroi = color.rgb2gray(imgroi)
                imgroi = cv2.resize(imgroi, (80,80))
                feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                feat = feat_hog.tolist()                
                feat_str = ""
                for value in feat:
                    feat_str = feat_str + str(value) + ','
                value_oneframe.append(feat_str)
                imgroi = img[oL_ymin:oL_ymax+1, oL_xmin:oL_xmax+1]
                imgroi = color.rgb2gray(imgroi)
                imgroi = cv2.resize(imgroi, (80,80))
                feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                feat = feat_hog.tolist()
                feat_str = ""
                for value in feat:
                    feat_str = feat_str + str(value) + ','
                value_oneframe.append(feat_str)
                if g_a_left.has_key(hL_grasp):
                    g_a_left[hL_grasp].append(value_oneframe)
                else:
                    g_a_left[hL_grasp] = []
                    g_a_left[hL_grasp].append(value_oneframe)
            if is_hRvisible == 1 and is_oRvisible == 1:
                value_oneframe = []
                value_oneframe.append(str(seqname)+","+str(filename)+",right")
                value_oneframe.append(str(is_oRprismatic)+","+str(is_oRsphere)+","+str(is_oRflat)+","+str(is_oRrigid))
                imgroi = img[hR_ymin:hR_ymax+1, hR_xmin:hR_xmax+1]
                imgroi = color.rgb2gray(imgroi)
                imgroi = cv2.resize(imgroi, (80,80))
                feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                feat = feat_hog.tolist()                
                feat_str = ""
                for value in feat:
                    feat_str = feat_str + str(value) + ','
                value_oneframe.append(feat_str)
                imgroi = img[oR_ymin:oR_ymax+1, oR_xmin:oR_xmax+1]
                imgroi = color.rgb2gray(imgroi)
                imgroi = cv2.resize(imgroi, (80,80))
                feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                feat = feat_hog.tolist()
                feat_str = ""
                for value in feat:
                    feat_str = feat_str + str(value) + ','
                value_oneframe.append(feat_str)
                if g_a_right.has_key(hR_grasp):
                    g_a_right[hR_grasp].append(value_oneframe)
                else:
                    g_a_right[hR_grasp] = []
                    g_a_right[hR_grasp].append(value_oneframe)
    #        break
    #    break 
    return g_a_left, g_a_right
    
#get pretrained network
LAYER = 'pool5'
INDEX = 4    
net = get_deep_net()
#dataset setting
img_dir = "/home/cai-mj/_GTA/img/GTEA"
seqs = []
seqs.append("006")
seqs.append("007")
seqs.append("008")
seqs.append("010")
seqs.append("012")
seqs.append("013")
seqs.append("014")
seqs.append("016")
seqs.append("017")
seqs.append("018")
seqs.append("021")
seqs.append("022")

seqs.append("002")
seqs.append("003")
seqs.append("005")
seqs.append("020")
grasp_trainfile = 'data/feature_grasp_train.csv'
grasp_testfile = 'data/feature_grasp_test.csv'
attribute_trainfile = 'data/feature_attribute_train.csv'
attribute_testfile = 'data/feature_attribute_test.csv'
[g_a_left, g_a_right] = extract_hog_grasp_attribute(seqs)
grasp_train = open(grasp_trainfile, 'w')
grasp_train.write("seqname,filename,side,grasp,feature...\n")
grasp_test = open(grasp_testfile, 'w')
grasp_test.write("seqname,filename,side,grasp,feature...\n")
attribute_train = open(attribute_trainfile, 'w')
attribute_train.write("seqname,filename,side,prismatic,sphere,flat,rigid,feature...\n")
attribute_test = open(attribute_testfile, 'w')
attribute_test.write("seqname,filename,side,prismatic,sphere,flat,rigid,feature...\n")
print [len(g_a_left), len(g_a_right)]
for grasp_type in g_a_left.keys():
    value_array = g_a_left[grasp_type]
    #write training data from 80% of all data
    for j in range(0, int(len(value_array)*4/5), 1):
        grasp_train.write(value_array[j][0]+","+grasp_type+","+value_array[j][2]+"\n")
        attribute_train.write(value_array[j][0]+","+value_array[j][1]+","+value_array[j][3]+"\n")
    #write test data from 20% of all data
    for j in range(int(len(value_array)*4/5), len(value_array), 1):
        grasp_test.write(value_array[j][0]+","+grasp_type+","+value_array[j][2]+"\n")
        attribute_test.write(value_array[j][0]+","+value_array[j][1]+","+value_array[j][3]+"\n")

for grasp_type in g_a_right.keys():
    value_array = g_a_right[grasp_type]
    #write training data from 80% of all data
    for j in range(0, int(len(value_array)*4/5), 1):
        grasp_train.write(value_array[j][0]+","+grasp_type+","+value_array[j][2]+"\n")
        attribute_train.write(value_array[j][0]+","+value_array[j][1]+","+value_array[j][3]+"\n")
    #write test data from 20% of all data
    for j in range(int(len(value_array)*4/5), len(value_array), 1):
        grasp_test.write(value_array[j][0]+","+grasp_type+","+value_array[j][2]+"\n")
        attribute_test.write(value_array[j][0]+","+value_array[j][1]+","+value_array[j][3]+"\n")
        
grasp_train.close()
grasp_test.close()
attribute_train.close()
attribute_test.close()
