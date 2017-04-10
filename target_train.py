# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:03:22 2015

@author: cai-mj
"""

import numpy as np
import os
import sys
import svmutil
import cv2
import yaml
import math

def read_info(filename, side):
    """
    read header info from feature.csv\n
    """
    f_feat = open(filename, "r")
    first_line = f_feat.readline()
    tags = first_line.split(',')
    idx_s = -1
    for i in range(len(tags)):
        if tags[i] == "side":
            idx_s = i
    assert(idx_s != -1)
    features = f_feat.readlines()
    f_feat.close()
    info = []
    for item in features:
        item_split = item.split(',')
        if side != item_split[idx_s]:
            continue
        info.append("_".join(item_split[0:idx_s+1]))
    return info
    
    
def write_svmdata_target(srcfile, datafile1, datafile2, datafile3, side):
    """
    write formed data required by libsvm\n
    """
    f_feat = open(srcfile, "r")
    first_line = f_feat.readline()
    tags = first_line.split(',')
    idx_rx = -1
    idx_ry = -1
    idx_rs = -1
    idx_f = -1
    idx_s = -1
    for i in range(len(tags)):
        if tags[i] == "relativeX":
            idx_rx = i
        if tags[i] == "relativeY":
            idx_ry = i
        if tags[i] == "relativeS":
            idx_rs = i
        if tags[i].find("feature") == 0:
            idx_f = i
        if tags[i] == "side":
            idx_s = i
    assert(idx_rx != -1 and idx_ry != -1 and idx_f != -1 and idx_s != -1)
    items = f_feat.readlines()
    f_feat.close()    
    rXs = []
    rYs = []
    rSs = []
    features = []   
    for item in items:
        item_split = item.split(',')
        if side != item_split[idx_s]:
            continue  
        rXs.append(item_split[idx_rx])
        rYs.append(item_split[idx_ry])
        rSs.append(item_split[idx_rs])
        features.append(item_split[idx_f:len(item_split)-1])
        
    #project to PC axis
    data = np.array(features, dtype=np.float32)
#    print "data before pca: ", data.shape
#    data = feature_pca(data)
#    print "data after pca: ", data.shape
      

    #write formed data required by libsvm
    #relativeX
    f_svmdata = open(datafile1, "w")
    for i in range(len(rXs)):
        f_svmdata.write(str(rXs[i])+" ")
        for j in range(data.shape[1]):
            f_svmdata.write(str(j+1)+":"+str(data[i,j])+" ")
        f_svmdata.write("\n")
    f_svmdata.close() 
    #relativeY    
    f_svmdata = open(datafile2, "w")
    for i in range(len(rYs)):
        f_svmdata.write(str(rYs[i])+" ")
        for j in range(data.shape[1]):
            f_svmdata.write(str(j+1)+":"+str(data[i,j])+" ")
        f_svmdata.write("\n")
    f_svmdata.close() 
    #relativeScale    
    f_svmdata = open(datafile3, "w")
    for i in range(len(rSs)):
        f_svmdata.write(str(rSs[i])+" ")
        for j in range(data.shape[1]):
            f_svmdata.write(str(j+1)+":"+str(data[i,j])+" ")
        f_svmdata.write("\n")
    f_svmdata.close() 
    

def train_target(side):
    """
    train_target(): 
    train linear svm regressor for target object location\n
    """
    #train
    datafile1 = "model/traindata_targetX_" + side
    datafile2 = "model/traindata_targetY_" + side
    datafile3 = "model/traindata_targetS_" + side
    if not os.path.isfile(datafile1):
        srcfile = "data/feature_target_train.csv"
        write_svmdata_target(srcfile, datafile1, datafile2, datafile3, side)    
    x_train,data_train = svmutil.svm_read_problem(datafile1)
    y_train,data_train = svmutil.svm_read_problem(datafile2)
    s_train,data_train = svmutil.svm_read_problem(datafile3)
    modelfile1 = "model/model_targetX_" + side
    modelfile2 = "model/model_targetY_" + side
    modelfile3 = "model/model_targetS_" + side
    m1 = []
    m2 = []
    m3 = []
    if not os.path.isfile(modelfile1):
        print("train model: " + side +"_targetX")
        prob = svmutil.svm_problem(x_train, data_train)
        param = svmutil.svm_parameter('-s 3 -t 0 -c 1 -p 0.05 -b 0 -q')
        m1 = svmutil.svm_train(prob, param)        
        svmutil.svm_save_model(modelfile1, m1)
        print("train model: " + side +"_targetY")
        prob = svmutil.svm_problem(y_train, data_train)
        param = svmutil.svm_parameter('-s 3 -t 0 -c 1 -p 0.05 -b 0 -q')
        m2 = svmutil.svm_train(prob, param)        
        svmutil.svm_save_model(modelfile2, m2)
        print("train model: " + side +"_targetS")
        prob = svmutil.svm_problem(s_train, data_train)
        param = svmutil.svm_parameter('-s 3 -t 0 -c 1 -p 0.05 -b 0 -q')
        m3 = svmutil.svm_train(prob, param)        
        svmutil.svm_save_model(modelfile3, m3)
    else:
        print("load model: " + side +"_targetX")
        m1 = svmutil.svm_load_model(modelfile1)
        print("load model: " + side +"_targetY")
        m2 = svmutil.svm_load_model(modelfile2)
        print("load model: " + side +"_targetS")
        m3 = svmutil.svm_load_model(modelfile3)
    #test    
    target_info = read_info("data/feature_target_test.csv", side)
    datafile1 = "model/testdata_targetX_" + side
    datafile2 = "model/testdata_targetY_" + side
    datafile3 = "model/testdata_targetS_" + side
    if not os.path.isfile(datafile1):
        srcfile = "data/feature_target_test.csv"
        write_svmdata_target(srcfile, datafile1, datafile2, datafile3, side)    
    x_test,data_test = svmutil.svm_read_problem(datafile1)
    y_test,data_test = svmutil.svm_read_problem(datafile2)
    s_test,data_test = svmutil.svm_read_problem(datafile3)
    #relativeX
    p_label, p_acc, p_val = svmutil.svm_predict(x_test, data_test, m1, '-b 0')
    f_result = open("result/targetX_" + side + ".csv", "w")
    for i in range(len(p_label)):
        f_result.write(target_info[i]+", "+str(x_test[i])+", "+str(p_label[i])+", ")
        f_result.write("\n")
    f_result.close()
    #relativeY
    p_label, p_acc, p_val = svmutil.svm_predict(y_test, data_test, m2, '-b 0')
    f_result = open("result/targetY_" + side + ".csv", "w")
    for i in range(len(p_label)):
        f_result.write(target_info[i]+", "+str(y_test[i])+", "+str(p_label[i])+", ")
        f_result.write("\n")
    f_result.close()
    #relativeScale
    p_label, p_acc, p_val = svmutil.svm_predict(s_test, data_test, m3, '-b 0')
    f_result = open("result/targetS_" + side + ".csv", "w")
    for i in range(len(p_label)):
        f_result.write(target_info[i]+", "+str(s_test[i])+", "+str(p_label[i])+", ")
        f_result.write("\n")
    f_result.close()
    

def drawCandidateBox(img, p_estimate, seq, frame):
    """
    draw candidate boxes from selective search results
    """
    PINK = [255,0,255]
    f_objectness = open("/home/cai-mj/programs/vision/SelectiveSearchCodeIJCV/"+seq+"/"+frame+"_box.csv", "r")
    lines = f_objectness.readlines()
    cdd = {}
    for line in lines:
        rect = line.split(",")
        bbox = [int(rect[i]) for i in range(4)]
        [width, height] = [bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]
        [cx, cy] = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        if p_estimate[0]<bbox[0] or p_estimate[0]>bbox[2] or p_estimate[1]<bbox[1] or p_estimate[1]>bbox[3]:
            continue
        if float(width)/height > 5 or float(height)/width > 5:
            continue
        if width > img.shape[1]/2 or height > img.shape[0]/2:
            continue
        if width < 20 or height < 20:
            continue
        dist = math.sqrt((cx-p_estimate[0])**2+(cy-p_estimate[1])**2)/math.sqrt(width*height)
        cdd[dist] = bbox
    sorted_dist = sorted(cdd)
    for i in range(min(3, len(sorted_dist))):
        bbox = cdd[sorted_dist[i]]
        cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), PINK, 1)
    
    
def visualize():
    """
    visualize the predicted location of target objects
    """
    RED = [0,0,255]
    GREEN = [0,255,0]
    BLUE = [255,0,0]
    img_dir = "/home/cai-mj/_GTA/img/GTEA"

    #read predicted location (relative scale)
    target_info_left = []
    rX_left = []
    f_result = open("result/targetX_" + "left" + ".csv", "r")
    lines = f_result.readlines()
    for line in lines:
        items = line.split(",")
        info = items[0].split("_")[0:2]
        target_info_left.append("_".join(info))
        rX_left.append(float(items[2]))
    f_result.close()
    rY_left = []
    f_result = open("result/targetY_" + "left" + ".csv", "r")
    lines = f_result.readlines()
    for line in lines:
        items = line.split(",")
        rY_left.append(float(items[2]))
    f_result.close()
    rS_left = []
    f_result = open("result/targetS_" + "left" + ".csv", "r")
    lines = f_result.readlines()
    for line in lines:
        items = line.split(",")
        rS_left.append(float(items[2]))
    f_result.close()
    target_info_right = []
    rX_right = []
    f_result = open("result/targetX_" + "right" + ".csv", "r")
    lines = f_result.readlines()
    for line in lines:
        items = line.split(",")
        info = items[0].split("_")[0:2]
        target_info_right.append("_".join(info))
        rX_right.append(float(items[2]))
    f_result.close()
    rY_right = []
    f_result = open("result/targetY_" + "right" + ".csv", "r")
    lines = f_result.readlines()
    for line in lines:
        items = line.split(",")
        rY_right.append(float(items[2]))
    f_result.close()
    rS_right = []
    f_result = open("result/targetS_" + "right" + ".csv", "r")
    lines = f_result.readlines()
    for line in lines:
        items = line.split(",")
        rS_right.append(float(items[2]))
    f_result.close()

    #draw visualization when both left/right hand are labeled
    for i in range(len(target_info_left)):
        for j in range(len(target_info_right)):
            if target_info_left[i] != target_info_right[j]:
                continue
            [seq, frame] = target_info_left[i].split("_")[0:2]
            f_yml = open(seq+"/"+frame+".yml", "r")
            yml_dict = yaml.load(f_yml.read())
            f_yml.close()
            is_oLvisible = yml_dict["leftobject"]["visible"]
            oL_xmin = yml_dict["leftobject"]["bndbox"]["xmin"]
            oL_ymin = yml_dict["leftobject"]["bndbox"]["ymin"]
            oL_xmax = yml_dict["leftobject"]["bndbox"]["xmax"]
            oL_ymax = yml_dict["leftobject"]["bndbox"]["ymax"]
            is_oRvisible = yml_dict["rightobject"]["visible"]
            oR_xmin = yml_dict["rightobject"]["bndbox"]["xmin"]
            oR_ymin = yml_dict["rightobject"]["bndbox"]["ymin"]
            oR_xmax = yml_dict["rightobject"]["bndbox"]["xmax"]
            oR_ymax = yml_dict["rightobject"]["bndbox"]["ymax"]
            is_hLvisible = yml_dict["lefthand"]["visible"]
            hL_xmin = yml_dict["lefthand"]["bndbox"]["xmin"]
            hL_ymin = yml_dict["lefthand"]["bndbox"]["ymin"]
            hL_xmax = yml_dict["lefthand"]["bndbox"]["xmax"]
            hL_ymax = yml_dict["lefthand"]["bndbox"]["ymax"]
            is_hRvisible = yml_dict["righthand"]["visible"]
            hR_xmin = yml_dict["righthand"]["bndbox"]["xmin"]
            hR_ymin = yml_dict["righthand"]["bndbox"]["ymin"]
            hR_xmax = yml_dict["righthand"]["bndbox"]["xmax"]
            hR_ymax = yml_dict["righthand"]["bndbox"]["ymax"]
            img = cv2.imread(img_dir+"/"+seq+"/"+frame+".jpg")
            if is_oLvisible == 1:
                cv2.rectangle(img, (oL_xmin,oL_ymin), (oL_xmax,oL_ymax), GREEN, 2)
                pX = rX_left[i]*(hL_xmax-hL_xmin+1) + (hL_xmax+hL_xmin)/2
                pY = rY_left[i]*(hL_ymax-hL_ymin+1) + (hL_ymax+hL_ymin)/2
                pS = rS_left[i]*math.sqrt((hL_xmax-hL_xmin+1)*(hL_ymax-hL_ymin+1))
                cv2.circle(img, (int(pX),int(pY)), int(pS/2), RED, thickness=2)
                #drawCandidateBox(img, (pX,pY), seq, frame)
            if is_oRvisible == 1:
                cv2.rectangle(img, (oR_xmin,oR_ymin), (oR_xmax,oR_ymax), GREEN, 2)
                pX = rX_right[j]*(hR_xmax-hR_xmin+1) + (hR_xmax+hR_xmin)/2
                pY = rY_right[j]*(hR_ymax-hR_ymin+1) + (hR_ymax+hR_ymin)/2
                pS = rS_right[i]*math.sqrt((hR_xmax-hR_xmin+1)*(hR_ymax-hR_ymin+1))
                cv2.circle(img, (int(pX),int(pY)), int(pS/2), RED, thickness=2)
                #drawCandidateBox(img, (pX,pY), seq, frame)
            if is_hLvisible == 1:
                cv2.rectangle(img, (hL_xmin,hL_ymin), (hL_xmax,hL_ymax), BLUE, 2)           
            if is_hRvisible == 1:
                cv2.rectangle(img, (hR_xmin,hR_ymin), (hR_xmax,hR_ymax), BLUE, 2)
            cv2.imwrite("result/target/"+seq+"_"+frame+".jpg", img)

    
def visualize_right():
    """
    visualize the predicted location of target objects
    """
    RED = [0,0,255]
    GREEN = [0,255,0]
    BLUE = [255,0,0]
    img_dir = "/home/cai-mj/_GTA/img/GTEA"
    #read predicted location (relative scale)
    target_info_right = []
    rX_right = []
    f_result = open("result/targetX_" + "right" + ".csv", "r")
    lines = f_result.readlines()
    for line in lines:
        items = line.split(",")
        info = items[0].split("_")[0:2]
        target_info_right.append("_".join(info))
        rX_right.append(float(items[2]))
    f_result.close()
    rY_right = []
    f_result = open("result/targetY_" + "right" + ".csv", "r")
    lines = f_result.readlines()
    for line in lines:
        items = line.split(",")
        rY_right.append(float(items[2]))
    f_result.close()
    #draw visualization when both left/right hand are labeled
    for i in range(len(target_info_right)):
            [seq, frame] = target_info_right[i].split("_")[0:2]
            f_yml = open(seq+"/"+frame+".yml", "r")
            yml_dict = yaml.load(f_yml.read())
            f_yml.close()
            is_oRvisible = yml_dict["rightobject"]["visible"]
            oR_xmin = yml_dict["rightobject"]["bndbox"]["xmin"]
            oR_ymin = yml_dict["rightobject"]["bndbox"]["ymin"]
            oR_xmax = yml_dict["rightobject"]["bndbox"]["xmax"]
            oR_ymax = yml_dict["rightobject"]["bndbox"]["ymax"]
            is_hRvisible = yml_dict["righthand"]["visible"]
            hR_xmin = yml_dict["righthand"]["bndbox"]["xmin"]
            hR_ymin = yml_dict["righthand"]["bndbox"]["ymin"]
            hR_xmax = yml_dict["righthand"]["bndbox"]["xmax"]
            hR_ymax = yml_dict["righthand"]["bndbox"]["ymax"]
            img = cv2.imread(img_dir+"/"+seq+"/"+frame+".jpg")
            if is_oRvisible == 1:
                cv2.rectangle(img, (oR_xmin,oR_ymin), (oR_xmax,oR_ymax), GREEN, 2)
                pX = rX_right[i]*(hR_xmax-hR_xmin+1) + (hR_xmax+hR_xmin)/2
                pY = rY_right[i]*(hR_ymax-hR_ymin+1) + (hR_ymax+hR_ymin)/2
                cv2.circle(img, (int(pX),int(pY)), 4, RED, thickness=-1)          
            if is_hRvisible == 1:
                cv2.rectangle(img, (hR_xmin,hR_ymin), (hR_xmax,hR_ymax), BLUE, 2)
            cv2.imwrite("result/target/"+seq+"_"+frame+".jpg", img)
            
            
def main(argv):
    """
    """    
    train_target("left")
    train_target("right")
#    visualize()
    
    
if __name__ == "__main__":
    main(sys.argv[1:])