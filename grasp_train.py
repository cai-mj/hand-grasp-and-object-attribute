# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:10:51 2015

@author: cai-mj
"""
import numpy as np
import os
import sys
import svmutil
import cv2
from common import read_info, test_eval, transfer_grasp_label

    
def get_freq_grasp(side, train_limit, test_limit):
    """
    get frequent grasp types and their occurrence frequence
    """
    train_file = "data/feature_grasp_train.csv"
    test_file = "data/feature_grasp_test.csv"
    f_feat = open(train_file, "r")
    first_line = f_feat.readline()
    tags = first_line.split(',')
    idx_g = -1
    idx_s = -1
    for i in range(len(tags)):
        if tags[i] == "grasp":
            idx_g = i
        if tags[i] == "side":
            idx_s = i
    assert(idx_g != -1 and idx_s != -1)
    grasp_count_train = {}
    features = f_feat.readlines()
    for item in features:
        item_split = item.split(',')
        if side != item_split[idx_s]:
            continue
        grasp_type = transfer_grasp_label(item_split[idx_g])
        if grasp_count_train.has_key(grasp_type):
            grasp_count_train[grasp_type] = grasp_count_train[grasp_type] + 1
        else:
            grasp_count_train[grasp_type] = 1
    for item in grasp_count_train.keys():
        if grasp_count_train[item] < train_limit:
            del grasp_count_train[item]
    f_feat.close()
    
    f_feat = open(test_file, "r")
    first_line = f_feat.readline()
    tags = first_line.split(',')
    idx_g = -1
    idx_s = -1
    for i in range(len(tags)):
        if tags[i] == "grasp":
            idx_g = i
        if tags[i] == "side":
            idx_s = i
    assert(idx_g != -1 and idx_s != -1)
    grasp_count_test = {}
    features = f_feat.readlines()
    for item in features:
        item_split = item.split(',')
        if side != item_split[idx_s]:
            continue
        grasp_type = transfer_grasp_label(item_split[idx_g])
        if grasp_count_test.has_key(grasp_type):
            grasp_count_test[grasp_type] = grasp_count_test[grasp_type] + 1
        else:
            grasp_count_test[grasp_type] = 1
    for item in grasp_count_test.keys():
        if grasp_count_test[item] < test_limit:
            del grasp_count_test[item]
    f_feat.close()
    return [grasp_count_train, grasp_count_test]
    

def train_pca(dimension=100):
    """
    Performs Principle Component Analysis of the training data and save mean, eigenvectors
    """
    print 'PCA for grasp training data...\n'
    train_file = "data/feature_grasp_train.csv"
    f_feat = open(train_file, "r")
    first_line = f_feat.readline()
    tags = first_line.split(',')
    idx_f = -1
    idx_s = -1
    for i in range(len(tags)):
        if tags[i].find("feature") == 0:
            idx_f = i
        if tags[i] == "side":
            idx_s = i
    assert(idx_f != -1 and idx_s != -1)
    features = f_feat.readlines()
    feature_listOfList = []
    for item in features:
        item_split = item.split(',')
        #only process right hand; modify this if double hands are to be process
#        if "left" == item_split[idx_s]:
#            continue
        feature_listOfList.append(item_split[idx_f:len(item_split)-1])
    dataMatrix = np.array(feature_listOfList, dtype=np.float32)
    print dataMatrix.shape, dataMatrix.dtype
    mean, eigenvectors = cv2.PCACompute(data=dataMatrix, maxComponents=dimension)
    print mean.shape, eigenvectors.shape
    
    return [mean, eigenvectors]
    
    
def feature_pca(feature):
    """
    get dimension-reduced feature by Principle Component Analysis\n
    feature: numpy array, each row is a data point
    """
    mean = np.array([])
    eigenvectors = np.array([])
    pca_file = "data/pca_grasp.npz"
    if not os.path.isfile(pca_file):
        mean, eigenvectors = train_pca()
        np.savez(pca_file, m=mean, v = eigenvectors)
    else:
        data = np.load(pca_file)
        mean = np.array(data['m'], data['m'].dtype)
        eigenvectors = np.array(data['v'], data['v'].dtype)
        data.close()
        print "load pca model: ", mean.shape, eigenvectors.shape
    result = cv2.PCAProject(feature, mean, eigenvectors)
    return result
    
        
def write_svmdata_grasp(srcfile, datafile, grasp_type, side, isTest):
    """
    write formed data required by libsvm\n
    """
    f_feat = open(srcfile, "r")
    first_line = f_feat.readline()
    tags = first_line.split(',')
    idx_g = -1
    idx_f = -1
    idx_s = -1
    for i in range(len(tags)):
        if tags[i] == "grasp":
            idx_g = i
        if tags[i].find("feature") == 0:
            idx_f = i
        if tags[i] == "side":
            idx_s = i
    assert(idx_g != -1 and idx_f != -1 and idx_s != -1)
    items = f_feat.readlines()
    f_feat.close()    
    labels = []
    features = []   
    for item in items:
        item_split = item.split(',')
        #only process right hand; modify this if double hands are to be process
        if side != item_split[idx_s]:
            continue       
        label = 0
        if grasp_type == transfer_grasp_label(item_split[idx_g]):
            label = 1
        labels.append(label)
        features.append(item_split[idx_f:len(item_split)-1])
        
    #project to PC axis
    data = np.array(features, dtype=np.float32)
#    print "data before pca: ", data.shape
#    data = feature_pca(data)
#    print "data after pca: ", data.shape
      
    #make sure the first data is positive sample
    if labels[0] < 1 and isTest == 0:
        idx_pos = -1
        for i in range(len(labels)):
            if labels[i] == 1:
                idx_pos = i
        assert(idx_pos != -1)
        data_temp = np.array(data[idx_pos,:], data.dtype)
        for j in range(data.shape[1]):
            data[idx_pos,j] = data[0,j]
            data[0,j] = data_temp[j]
        labels[idx_pos] = labels[0]
        labels[0] = 1
    #write formed data required by libsvm
    f_svmdata = open(datafile, "w")
    for i in range(len(labels)):
        f_svmdata.write(str(labels[i])+" ")
        for j in range(data.shape[1]):
            f_svmdata.write(str(j+1)+":"+str(data[i,j])+" ")
        f_svmdata.write("\n")
    f_svmdata.close() 
    

def train_grasp(grasp_type, side):
    """
    train_grasp(grasp_type): 
    train linear svm classifier for specific grasp type\n
    grasp_type: hand grasping type\n
    side: left hand or right hand\n
    """
    #train
    datafile = "model/traindata_grasp_"+grasp_type+"_"+side
    if not os.path.isfile(datafile):
        srcfile = "data/feature_grasp_train.csv"
        write_svmdata_grasp(srcfile, datafile, grasp_type, side, 0)    
    label_train,data_train = svmutil.svm_read_problem(datafile)
    modelfile = "model/model_grasp_"+grasp_type+"_"+side
    m = []
    if not os.path.isfile(modelfile):
        print("train model: " + grasp_type + "_" + side)
        label_weight = {}
        for v in label_train:
            if label_weight.has_key(v):
                label_weight[v]+=1
            else:
                label_weight[v]=1
        sorted_label = sorted(label_weight)
        param_weight = ' '
        for v in sorted_label:
            label_weight[v] = float(len(label_train))/len(sorted_label)/label_weight[v]
            param_weight += '-w%d %f ' % (v, label_weight[v])
        prob = svmutil.svm_problem(label_train, data_train)
        param = svmutil.svm_parameter('-t 0 -b 1 -q'+param_weight)
        print '-t 0 -b 1 -q'+param_weight
#        param = svmutil.svm_parameter('-t 0 -c 4 -b 1 -q')
        m = svmutil.svm_train(prob, param)        
        svmutil.svm_save_model(modelfile, m)
    else:
        print("load model: " + grasp_type + "_" + side)
        m = svmutil.svm_load_model(modelfile)
    #test    
    grasp_info = read_info("data/feature_grasp_test.csv", side)
    datafile = "model/testdata_grasp_"+grasp_type+"_"+side
    if not os.path.isfile(datafile):
        srcfile = "data/feature_grasp_test.csv"
        write_svmdata_grasp(srcfile, datafile, grasp_type, side, 1)    
    label_test,data_test = svmutil.svm_read_problem(datafile)
    p_label, p_acc, p_val = svmutil.svm_predict(label_test, data_test, m, '-b 1')
    f_result = open("result/grasp_" + grasp_type + "_" + side + ".csv", "w")
    for i in range(len(p_label)):
        f_result.write(grasp_info[i]+", "+str(int(label_test[i]))+", "+str(int(p_label[i]))+", ")
        f_result.write("[%.4f]\n" % p_val[i][0])
    f_result.close()
    
    
def test_multi_grasp(grasp_freq, side):
    """
    predict grasp type with highest score from multiple models
    """
    grasp_info = []
    scores = []
    labels = []
    for grasp_type in grasp_freq:
        resultfile = "result/grasp_" + grasp_type + "_" + side + ".csv"
        if not os.path.isfile(resultfile):
            print "can't open result file: "+resultfile, " (please train your models)\n"
            return
        f_result = open(resultfile, "r")
        lines = f_result.readlines()
        seqframe = []
        score_of_one_model = []
        label_of_one_model = []
        for item in lines:
            label_start = item.find(", ")
            seqframe.append(item[0:label_start])
            label_end = item.find(",", label_start+1)
            label_of_one_model.append(int(float(item[label_start+1:label_end])))
            score_start = item.find("[")
            score_end = item.find("]", score_start+1)
            score_of_one_model.append(float(item[score_start+1:score_end]))
        scores.append(score_of_one_model)
        labels.append(label_of_one_model)
        grasp_info = seqframe
    score_array = np.array(scores, np.float32).transpose()
    label_array = np.array(labels, np.int32).transpose()
    label_gt = np.argmax(label_array, axis=1)
    label_predict = np.argmax(score_array, axis=1) #return indices of the maximum along column axis
    del_index = [] #for some case when the label doesnot belong to trained classes
    for (index, value) in enumerate(label_gt):
        if label_array[index, value] < 1:
            del_index.append(index)
    label_gt = np.delete(label_gt, del_index)
    label_predict = np.delete(label_predict, del_index)
    for i in range(len(del_index)-1, -1, -1):
        del grasp_info[del_index[i]]
    test_eval(label_gt, label_predict, grasp_freq)  
    f_result = open("result/grasp_all_"+side+".csv", "w")
    f_result.write("info, label, predict, score...\n")
    for i in range(label_gt.size):
        f_result.write(grasp_info[i]+", "+str(int(label_gt[i]))+", "+str(int(label_predict[i]))+", ")
        for j in range(score_array.shape[1]):
            f_result.write("%.4f, " % score_array[i,j])
        f_result.write("\n")
    f_result.close()
        
def main(argv):
    """
    """   
    print "left hand grasp==>"
    grasp_count_train, grasp_count_test = get_freq_grasp("left", 3, 1)
    grasp_freq = []
    sorted_type = sorted(grasp_count_train)
    for item in sorted_type:
        if grasp_count_test.has_key(item):
            grasp_freq.append(item)
    for item in grasp_freq:
        print item, grasp_count_train[item], grasp_count_test[item]
        train_grasp(item, "left")
    test_multi_grasp(grasp_freq, "left")
    print "right hand grasp==>"
    grasp_count_train, grasp_count_test = get_freq_grasp("right", 3, 1)
    grasp_freq = []
    sorted_type = sorted(grasp_count_train)
    for item in sorted_type:
        if grasp_count_test.has_key(item):
            grasp_freq.append(item)
    for item in grasp_freq:
        print item, grasp_count_train[item], grasp_count_test[item]
        train_grasp(item, "right")
    test_multi_grasp(grasp_freq, "right")

    
    
if __name__ == "__main__":
    main(sys.argv[1:])