# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:25:01 2015

@author: cai-mj
"""

import numpy as np
import os
import yaml
import caffe

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


def overlap_rate(box1, box2):
    """
    division of intersection area by union area of two rectangle region\n
    box1,box2: list of four corner points
    """
    overlap_region = [max(box1[0],box2[0]),max(box1[1],box2[1]),min(box1[2],box2[2]),min(box1[3],box2[3])]
    if overlap_region[2]-overlap_region[0]+1>0 and overlap_region[3]-overlap_region[1]+1>0:
        overlap_area = (overlap_region[2]-overlap_region[0]+1)*(overlap_region[3]-overlap_region[1]+1)
    else:
        overlap_area = 0
    union_area = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)+(box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
    return overlap_area/union_area


def extract_cnn_attribute(seqs, dstfile):
    """
    extract deep convolutional network features for object attributes\n
    seqs: list of sequence names\n
    dstfile: file to which features are written
    """
    print('Prepare cnn feature to ' + dstfile + '\n')
    myfile = open(dstfile, 'w')
    myfile.write("seqname,filename,side,prismatic,sphere,flat,rigid,feature...\n")
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
    #        print yml_dict
            seqname = str(yml_dict["seqname"])
            filename = str(yml_dict["filename"])
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
            if is_oLvisible == 1:
                imgroi = img[oL_ymin:oL_ymax+1, oL_xmin:oL_xmax+1]
                net.predict([imgroi])
                feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
                myfile.write(str(seqname)+","+str(filename)+",left,")
                myfile.write(str(is_oLprismatic)+",")
                myfile.write(str(is_oLsphere)+",")
                myfile.write(str(is_oLflat)+",")
                myfile.write(str(is_oLrigid)+",")
                for value in feat:
                    myfile.write(str(value) + ',')
                myfile.write('\n')
            if is_oRvisible == 1:
                imgroi = img[oR_ymin:oR_ymax+1, oR_xmin:oR_xmax+1]
                net.predict([imgroi])
                feat = net.blobs[LAYER].data[INDEX].flatten().tolist()
                myfile.write(str(seqname)+","+str(filename)+",right,")
                myfile.write(str(is_oRprismatic)+",")
                myfile.write(str(is_oRsphere)+",")
                myfile.write(str(is_oRflat)+",")
                myfile.write(str(is_oRrigid)+",")
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
#dataset
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
trainfile = 'data/feature_attribute_train.csv'
#testfile = 'data/feature_attribute_test.csv'
extract_cnn_attribute(seqs_train, trainfile)
#extract_cnn_attribute(seqs_test, testfile)
