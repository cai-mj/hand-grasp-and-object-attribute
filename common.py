# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:12:21 2015

@author: cai-mj
"""
import numpy as np
import os
import cv2

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
        info.append(item_split[0]+"_"+item_split[1]+"_"+item_split[2])
    return info
 
def read_model_linearSVM(filename, dim):
    """
    """
    alphas = []
    supvecs = []
    f_model = open(filename, "r") 
    line = f_model.readline() #svm_type
    line = f_model.readline() #kernel_type
    line = f_model.readline() #nr_class
    line = f_model.readline() #total_sv
    tags = line.split(" ")
    nsv = int(tags[1])
    line = f_model.readline() #rho
    tags = line.split(" ")
    bias = float(tags[1])
    line = f_model.readline() #label
    tags = line.split(" ")
    label = int(tags[1])
    line = f_model.readline() #probA
    line = f_model.readline() #probB
    line = f_model.readline() #nr_sv
    line = f_model.readline() #SV
    for i in range(nsv):
        line = f_model.readline()
        tags = line.split(" ")
        alphas.append(float(tags[0]))
        supvec = [0]*dim
        for j in range(1, len(tags)-1, 1):
            items = tags[j].split(":")
            supvec[items[0]] = float(items[1])
        supvecs.append(supvec)
    weight = [0]*len(supvecs[0])
    print "nsv: ", nsv, "number of supvecs:", len(supvecs)," length of supvec: ", len(supvecs[0])
    for i in range(nsv):
        alpha = alphas[i]
        vec = supvecs[i]
        for j in range(len(vec)):
            weight[j] += alpha*vec[j]
    if label < 1:
        weight = [v*(-1) for v in weight]
    return weight
    
def transfer_grasp_label(pre_type):
    """
    transfer grasp label to match the defined types required.
    """
    post_type = pre_type
    if pre_type == "large-diameter":
        post_type = "large-wrap"
    if pre_type == "light-tool":
        post_type = "small-wrap"
    if pre_type == "medium-wrap":
        post_type = "small-wrap"        
    if pre_type == "inferior-pincer":
        post_type = "precision-sphere"
    if pre_type == "tripod":
        post_type = "precision-sphere"
    if pre_type == "precision-disk":
        post_type = "precision-sphere"    
    
    if pre_type == "thumb-2-finger":
        post_type = "thumb-n-finger"
    if pre_type == "thumb-3-finger":
        post_type = "thumb-n-finger"
    if pre_type == "thumb-4-finger":
        post_type = "thumb-n-finger"
    if pre_type == "thumb-index-finger":
        post_type = "thumb-n-finger"
        
    if pre_type == "lateral-tripod":
        post_type = "lateral-pinch"
        
    return post_type
    
def test_eval(label_g, label_p, category):
    """
    evaluate classification result and output confusion matrix, accuracy
    """
#    print "shape of label_g:", label_g.shape, "shape of label_p:", label_p.shape
    num_class = len(category)
    assert(label_g.size == label_p.size and label_g[np.argmax(label_g)] == num_class-1)
    confusion = np.zeros((num_class,num_class), np.int32)
    for i in range(label_g.size):
        confusion[label_g[i],label_p[i]] = confusion[label_g[i],label_p[i]] + 1
    accuracy = np.trace(confusion, dtype=np.float32)/np.sum(confusion, dtype=np.float32)
    precision = []
    for i in range(confusion.shape[1]):
        precision.append(confusion[i,i]/(np.sum(confusion[:,i], dtype=np.float32)+0.001))
    recall = []
    f1 = []
    fraction = []
    for i in range(confusion.shape[0]):
        recall.append(confusion[i,i]/(np.sum(confusion[i,:], dtype=np.float32)+0.001))
        f1.append(2*precision[i]*recall[i]/(precision[i]+recall[i]+0.001))
        fraction.append(np.sum(confusion[i,:], dtype=np.float32)/np.sum(confusion, dtype=np.float32))
    print confusion
    for i in range(len(category)):
        print category[i], "[%.4f,%.4f,%.4f,%.4f]" % (fraction[i],precision[i],recall[i],f1[i])
    print "accuracy: ", accuracy
    #save confusion matrix
    f_conf = open("confusion.csv", 'w')
    for i in range(confusion.shape[0]):
        line_sum = np.sum(confusion[i,:], dtype=np.float32)+0.001
        for j in range(confusion.shape[1]):
            item = confusion[i,j]/line_sum
            f_conf.write("%.4f," % item)
        f_conf.write("\n")
    f_conf.close()
    
    
def train_pca(train_file, dimension=100):
    """
    Performs Principle Component Analysis of the training data and save mean, eigenvectors
    """
    print 'PCA for grasp training data...\n'
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
        if "left" == item_split[idx_s]:
            continue
        feature_listOfList.append(item_split[idx_f:len(item_split)-1])
    dataMatrix = np.array(feature_listOfList, dtype=np.float32)
    print dataMatrix.shape, dataMatrix.dtype
    mean, eigenvectors = cv2.PCACompute(data=dataMatrix, maxComponents=dimension)
    print mean.shape, eigenvectors.shape
    
    return [mean, eigenvectors]
    
    
def feature_pca(feature, pca_file, train_file):
    """
    get dimension-reduced feature by Principle Component Analysis\n
    feature: numpy array, each row is a data point
    """
    mean = np.array([])
    eigenvectors = np.array([])
    if not os.path.isfile(pca_file):
        mean, eigenvectors = train_pca(train_file)
        np.savez(pca_file, m=mean, v = eigenvectors)
    else:
        data = np.load(pca_file)
        mean = np.array(data['m'], data['m'].dtype)
        eigenvectors = np.array(data['v'], data['v'].dtype)
        data.close()
        print "load pca model: ", mean.shape, eigenvectors.shape
    result = cv2.PCAProject(feature, mean, eigenvectors)
    return result
    
    
def data_structural_svm(freq_grasps, attributes):
    """
    prepare data for structural svm
    """
    #data from grasps
    srcfile = "data/feature_grasp_train.csv"
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
    info_grasp = []
    labels_grasp = []
    features_grasp = []   
    for item in items:
        item_split = item.split(',')
        #only process right hand; modify this if double hands are to be process
        if "left" == item_split[idx_s]:
            continue       
        label = -1
        for (index, grasp) in enumerate(freq_grasps):
            if item_split[idx_g] == grasp:
                label = index
        if label == -1:
            continue
        labels_grasp.append(label)
        features_grasp.append(item_split[idx_f:len(item_split)-1])
        info_grasp.append(item_split[0]+"_"+item_split[1]+"_"+item_split[2])
        
    #project to PC axis
    pca_file = "data/pca_grasp.npz"
    data_grasp = np.array(features_grasp, dtype=np.float32)
    data_grasp = feature_pca(data_grasp, pca_file, srcfile)
    assert(len(labels_grasp) == len(info_grasp))
    
    #data from attributes
    srcfile = "data/feature_attribute_train.csv"
    f_feat = open(srcfile, "r")
    first_line = f_feat.readline()
    tags = first_line.split(',')
    idx_a = [-1 for t in attributes]
    idx_s = -1
    idx_f = -1
    for i in range(len(tags)):
        for (index, attribute) in enumerate(attributes):
            if tags[i] == attribute:
                idx_a[index] = i
        if tags[i].find("feature") == 0:
            idx_f = i
        if tags[i] == "side":
            idx_s = i
    for v in idx_a:
        assert(v != -1)
    assert(idx_f != -1 and idx_s != -1)
    items = f_feat.readlines()
    f_feat.close()
    labels_attribute = []
    features_attribute = [] 
    info_attribute = []
    for item in items:
        item_split = item.split(',')
        #only process right hand; modify this if double hands are to be process
        if "left" == item_split[idx_s]:
            continue       
        label = [int(item_split[v]) for v in idx_a]
        for (index, value) in enumerate(label):
            if value < 1:
                label[index] = -1
        labels_attribute.append(label)
        features_attribute.append(item_split[idx_f:len(item_split)-1])
        info_attribute.append(item_split[0]+"_"+item_split[1]+"_"+item_split[2])
        
    #project to PC axis
    pca_file = "data/pca_attribute.npz"
    data_attribute = np.array(features_attribute, dtype=np.float32)
    data_attribute = feature_pca(data_attribute, pca_file, srcfile)
    assert(len(labels_attribute) == len(info_attribute))

    #write datafile for joint feature and labels 
    f_joint = open("result/joint_feature.txt", "w") 
    for i in range(len(info_attribute)):
        idx_g = -1
        for j in range(len(info_grasp)):
            if info_attribute[i] == info_grasp[j]:
                idx_g = j
                break
        if idx_g == -1:
            print "WARNING: no matched info for: "+info_attribute[i]
            continue
        f_joint.write(info_attribute[i]+" ")
        for v in labels_attribute[i]:
            f_joint.write(str(v)+" ")
        f_joint.write(str(labels_grasp[idx_g])+" ")
        idx = 1
        for j in range(data_attribute.shape[1]):
            f_joint.write(str(idx)+":"+str(data_attribute[i,j])+" ")
            idx = idx+1
        for j in range(data_grasp.shape[1]):
            f_joint.write(str(idx)+":"+str(data_grasp[idx_g,j])+" ")
            idx = idx+1
        f_joint.write("\n")
    f_joint.close()
    
    
def run():
    """
    """
    attributes = []
    attributes.append("flat")
    attributes.append("prismatic")
    attributes.append("rigid")
    attributes.append("sphere")
    from grasp_train import get_freq_grasp
    grasp_count_train, grasp_count_test = get_freq_grasp(3, 1)
    freq_grasps = []
    sorted_type = sorted(grasp_count_train)
    for item in sorted_type:
        if grasp_count_test.has_key(item):
            freq_grasps.append(item)
    data_structural_svm(freq_grasps, attributes)
    
    
    
    