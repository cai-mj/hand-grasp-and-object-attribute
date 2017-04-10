# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:55:08 2015

@author: cai-mj
"""
import numpy as np
import os
import sys
import math
import yaml
import svmutil
import cv2
import caffe
from common import read_info


def get_deep_net():
    """ 
    get pretrained deep network model 
    """
    caffe_root = '/home/cai-mj/programs/caffe-master'
    MEAN_FILE = caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    MODEL_FILE = caffe_root + '/python/feature/imgnet_feature.prototxt'
    PRETRAINED = caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'        
    net = caffe.Classifier(MODEL_FILE, PRETRAINED, gpu=True)
    caffe.set_phase_test()
    net.set_mean('data', np.load(MEAN_FILE))
    net.set_raw_scale('data', 255)
    net.set_channel_swap('data', (2,1,0))
    return net
    
    
def get_attribute_freq(attribute, side):
    """
    get occurrence frequence of object attributes
    """
    train_file = "data/feature_attribute_train.csv"
    test_file = "data/feature_attribute_test.csv"
    f_feat = open(train_file, "r")
    first_line = f_feat.readline()
    tags = first_line.split(',')
    idx_a = -1
    idx_s = -1
    for i in range(len(tags)):
        if tags[i] == attribute:
            idx_a = i
        if tags[i] == "side":
            idx_s = i
    assert(idx_a != -1 and idx_s != -1)
    features = f_feat.readlines()
    f_feat.close()
    num_sample = 0
    num_pos = 0
    for item in features:
        item_split = item.split(',')
        if side != item_split[idx_s]:
            continue
        num_sample = num_sample + 1
        if int(item_split[idx_a]) == 1:
            num_pos = num_pos + 1
    train_rate = float(num_pos)/num_sample
    #print "pos:neg in training for", attribute+":", num_pos, ":", num_sample-num_pos

    f_feat = open(test_file, "r")
    first_line = f_feat.readline()
    tags = first_line.split(',')
    idx_a = -1
    idx_s = -1
    for i in range(len(tags)):
        if tags[i] == attribute:
            idx_a = i
        if tags[i] == "side":
            idx_s = i
    assert(idx_a != -1 and idx_s != -1)
    features = f_feat.readlines()
    f_feat.close()
    num_sample = 0
    num_pos = 0
    for item in features:
        item_split = item.split(',')
        if side != item_split[idx_s]:
            continue
        num_sample = num_sample + 1
        if int(item_split[idx_a]) == 1:
            num_pos = num_pos + 1
    test_rate = float(num_pos)/num_sample
    #print "pos:neg in test for", attribute+":", num_pos, ":", num_sample-num_pos
    return [train_rate, test_rate]
        

def train_pca(dimension=100):
    """
    Performs Principle Component Analysis of the training data and save mean, eigenvectors
    """
    print 'PCA for attribute training data...\n'
    train_file = "data/feature_attribute_train.csv"
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
    pca_file = "data/pca_attribute.npz"
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
    
    
def write_svmdata_attribute(srcfile, datafile, attribute, side, isTest):
    """
    write formed data required by libsvm\n
    srcfile: feature file of object attributes\n
    datafile: libsvm data file
    """
    f_feat = open(srcfile, "r")
    first_line = f_feat.readline()
    tags = first_line.split(',')
    idx_a = -1
    idx_s = -1
    idx_f = -1
    for i in range(len(tags)):
        if tags[i] == attribute:
            idx_a = i
        if tags[i].find("feature") == 0:
            idx_f = i
        if tags[i] == "side":
            idx_s = i
    assert(idx_a != -1 and idx_f != -1 and idx_s != -1)
    items = f_feat.readlines()
    f_feat.close()
    labels = []
    features = []   
    for item in items:
        item_split = item.split(',')
        if side != item_split[idx_s]:
            continue       
        label = int(item_split[idx_a])
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
    

def getF1(label, predict):  
    """
    """
    sum_tp = 0 #number of true positive
    sum_pl = 0 #number of positive label
    sum_pp = 0 #number of positive predict
    sum_ac = 0 #number of accurate prediction
    for i in range(len(label)):
        if label[i] == 1:
            sum_pl+=1
        if predict[i] == 1:
            sum_pp+=1
        if label[i] == 1 and predict[i] == 1:
            sum_tp+=1
        if label[i] == predict[i]:
            sum_ac+=1
    precision = sum_tp/(sum_pp+0.001)
    recall = sum_tp/(sum_pl+0.001)
    f1 = 2*precision*recall/(precision+recall+0.001)
    accuracy = sum_ac*1.0/len(label)
    return [precision, recall, f1, accuracy]    


def load_pretrained_targetmodels():
    """
    """
    #left target model
    m_target_l = []
    modelfile = "model/model_targetX_left"
    m = svmutil.svm_load_model(modelfile)
    m_target_l.append(m)
    modelfile = "model/model_targetY_left"
    m = svmutil.svm_load_model(modelfile)
    m_target_l.append(m)
    modelfile = "model/model_targetS_left"
    m = svmutil.svm_load_model(modelfile)
    m_target_l.append(m)
    #right target model
    m_target_r = []
    modelfile = "model/model_targetX_right"
    m = svmutil.svm_load_model(modelfile)
    m_target_r.append(m)
    modelfile = "model/model_targetY_right"
    m = svmutil.svm_load_model(modelfile)
    m_target_r.append(m)
    modelfile = "model/model_targetS_right"
    m = svmutil.svm_load_model(modelfile)
    m_target_r.append(m)
    return [m_target_l, m_target_r]
    

def get_feature_by_detection(seqs, dstfile):
    """
    """
    [m_target_l, m_target_r] = load_pretrained_targetmodels()
    #get pretrained network
    LAYER = 'fc6wi'
    LAYER_T = 'pool5'
    INDEX = 4    
    net = get_deep_net()
    img_dir = "/home/cai-mj/_GTA/img/GTEA"
    f_feat = open(dstfile, 'w')
    f_feat.write("seqname,filename,side,prismatic,sphere,flat,rigid,feature...\n")
    for seq in seqs:
        print("extract feature for: "+seq)
        command = "ls " + seq + "/*.yml > filename.txt"
        os.system(command)
        f_files = open("filename.txt", "r")
        files = f_files.readlines()
        f_files.close()  
        feats_tl = []
        feats_tr = []
        boxes_hl = []
        boxes_hr = []
        print("  grasp cnn feature...")
        for j in range(len(files)):
            #read from each yaml file
            ymlfile = files[j][:len(files[j])-1]
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
            is_hRvisible = yml_dict["righthand"]["visible"]
            hR_xmin = yml_dict["righthand"]["bndbox"]["xmin"]
            hR_ymin = yml_dict["righthand"]["bndbox"]["ymin"]
            hR_xmax = yml_dict["righthand"]["bndbox"]["xmax"]
            hR_ymax = yml_dict["righthand"]["bndbox"]["ymax"]
            is_oLvisible = yml_dict["leftobject"]["visible"]
            is_oRvisible = yml_dict["rightobject"]["visible"]
            img = caffe.io.load_image(img_dir+"/"+seqname+"/"+filename+".jpg")
            if is_hLvisible == 1 and is_oLvisible == 1:
                imgroi = img[hL_ymin:hL_ymax+1, hL_xmin:hL_xmax+1]
                net.predict([imgroi])
                feat_tl = net.blobs[LAYER_T].data[INDEX].flatten().tolist()
                feats_tl.append(feat_tl)
                boxes_hl.append([hL_xmin,hL_ymin,hL_xmax,hL_ymax])
            if is_hRvisible == 1 and is_oRvisible == 1:
                imgroi = img[hR_ymin:hR_ymax+1, hR_xmin:hR_xmax+1]
                net.predict([imgroi])
                feat_tr = net.blobs[LAYER_T].data[INDEX].flatten().tolist()
                feats_tr.append(feat_tr)
                boxes_hr.append([hR_xmin,hR_ymin,hR_xmax,hR_ymax])
        print("  predict object bounding box...")
        label = [0]*len(feats_tl)
        rX_left, p_acc, p_val = svmutil.svm_predict(label, feats_tl, m_target_l[0], '-b 0')
        rY_left, p_acc, p_val = svmutil.svm_predict(label, feats_tl, m_target_l[1], '-b 0')
        rS_left, p_acc, p_val = svmutil.svm_predict(label, feats_tl, m_target_l[2], '-b 0')
        label = [0]*len(feats_tr)
        rX_right, p_acc, p_val = svmutil.svm_predict(label, feats_tr, m_target_r[0], '-b 0')
        rY_right, p_acc, p_val = svmutil.svm_predict(label, feats_tr, m_target_r[1], '-b 0')
        rS_right, p_acc, p_val = svmutil.svm_predict(label, feats_tr, m_target_r[2], '-b 0')
        feats_ol = []
        feats_or = []
        info_left = []
        info_right = []
        label_left = []
        label_right = []
        print("  object cnn feature...")
        for j in range(len(files)):
            #read from each yaml file
            ymlfile = files[j][:len(files[j])-1]
            f_yml = open(ymlfile, "r")
            yml_dict = yaml.load(f_yml.read())
            f_yml.close()
            seqname = str(yml_dict["seqname"])
            filename = str(yml_dict["filename"])
            is_hLvisible = yml_dict["lefthand"]["visible"]
            is_hRvisible = yml_dict["righthand"]["visible"]
            is_oLvisible = yml_dict["leftobject"]["visible"]
            is_oLprismatic = yml_dict["leftobject"]["attribute"]["prismatic"]
            is_oLsphere = yml_dict["leftobject"]["attribute"]["sphere"]
            is_oLflat = yml_dict["leftobject"]["attribute"]["flat"]
            is_oLrigid = yml_dict["leftobject"]["attribute"]["rigid"]
            is_oRvisible = yml_dict["rightobject"]["visible"]
            is_oRprismatic = yml_dict["rightobject"]["attribute"]["prismatic"]
            is_oRsphere = yml_dict["rightobject"]["attribute"]["sphere"]
            is_oRflat = yml_dict["rightobject"]["attribute"]["flat"]
            is_oRrigid = yml_dict["rightobject"]["attribute"]["rigid"]
            img = caffe.io.load_image(img_dir+"/"+seqname+"/"+filename+".jpg")
            if is_hLvisible == 1 and is_oLvisible == 1:
                [hL_xmin,hL_ymin,hL_xmax,hL_ymax] = boxes_hl[len(feats_ol)]
                pX_left = rX_left[len(feats_ol)]*(hL_xmax-hL_xmin+1) + (hL_xmax+hL_xmin)/2
                pY_left = rY_left[len(feats_ol)]*(hL_ymax-hL_ymin+1) + (hL_ymax+hL_ymin)/2
                pS_left = rS_left[len(feats_ol)]*math.sqrt((hL_xmax-hL_xmin+1)*(hL_ymax-hL_ymin+1))
                [oL_xmin,oL_ymin,oL_xmax,oL_ymax] = [pX_left-pS_left/2,pY_left-pS_left/2,pX_left+pS_left/2,pY_left+pS_left/2]
                imgroi = img[oL_ymin:oL_ymax+1, oL_xmin:oL_xmax+1]
                net.predict([imgroi])
                feat_ol = net.blobs[LAYER].data[INDEX].flatten().tolist()
                feats_ol.append(feat_ol)
                info_left.append([seqname,filename,"left"])
                label_left.append([str(is_oLprismatic),str(is_oLsphere),str(is_oLflat),str(is_oLrigid)])
            if is_hRvisible == 1 and is_oRvisible == 1:
                [hR_xmin,hR_ymin,hR_xmax,hR_ymax] = boxes_hr[len(feats_or)]
                pX_right = rX_right[len(feats_or)]*(hR_xmax-hR_xmin+1) + (hR_xmax+hR_xmin)/2
                pY_right = rY_right[len(feats_or)]*(hR_ymax-hR_ymin+1) + (hR_ymax+hR_ymin)/2
                pS_right = rS_right[len(feats_or)]*math.sqrt((hR_xmax-hR_xmin+1)*(hR_ymax-hR_ymin+1))
                [oR_xmin,oR_ymin,oR_xmax,oR_ymax] = [pX_right-pS_right/2,pY_right-pS_right/2,pX_right+pS_right/2,pY_right+pS_right/2]
                imgroi = img[oR_ymin:oR_ymax+1, oR_xmin:oR_xmax+1]
                net.predict([imgroi])
                feat_or = net.blobs[LAYER].data[INDEX].flatten().tolist()
                feats_or.append(feat_or)
                info_right.append([seqname,filename,"right"])
                label_right.append([str(is_oRprismatic),str(is_oRsphere),str(is_oRflat),str(is_oRrigid)])
        print "  save features..."
        for j in range(len(feats_ol)):
            for k in range(len(info_left[j])):
                f_feat.write(info_left[j][k]+",")
            for k in range(len(label_left[j])):
                f_feat.write(label_left[j][k]+",")
            for k in range(len(feats_ol[j])):
                f_feat.write(str(feats_ol[j][k])+",")
            f_feat.write("\n")
        for j in range(len(feats_or)):
            for k in range(len(info_right[j])):
                f_feat.write(info_right[j][k]+",")
            for k in range(len(label_right[j])):
                f_feat.write(label_right[j][k]+",")
            for k in range(len(feats_or[j])):
                f_feat.write(str(feats_or[j][k])+",")
            f_feat.write("\n")
    f_feat.close()

   
def train_attribute(attribute, side):
    """
    train_attribute(str, float): 
    train linear svm classifier for specific attribute\n
    attribute: should be one from ["prismatic", "sphere", "flat", "rigid"]
    """
    #train
    datafile = "model/traindata_attribute_"+attribute+"_"+side
    if not os.path.isfile(datafile):
        srcfile = "data/feature_attribute_train.csv"
        write_svmdata_attribute(srcfile, datafile, attribute, side, 0)    
    label_train,data_train = svmutil.svm_read_problem(datafile)    
    modelfile = "model/model_attribute_"+attribute+"_"+side
    m = []
    if not os.path.isfile(modelfile):
        print("train model: " + attribute+"_"+side)
        prob = svmutil.svm_problem(label_train, data_train)
        param = svmutil.svm_parameter('-t 0 -c 4 -b 1 -q')
        m = svmutil.svm_train(prob, param)        
        svmutil.svm_save_model(modelfile, m)
    else:
        print("load model: " + attribute+"_"+side)
        m = svmutil.svm_load_model(modelfile)
    #test
    attribute_info = read_info("data/feature_attribute_test.csv", side)
    datafile = "model/testdata_attribute_"+attribute+"_"+side
    if not os.path.isfile(datafile):
        srcfile = "data/feature_attribute_test.csv"
        write_svmdata_attribute(srcfile, datafile, attribute, side, 1)    
    label_test,data_test = svmutil.svm_read_problem(datafile)
    p_label, p_acc, p_val = svmutil.svm_predict(label_test, data_test, m, '-b 1')
    [precision, recall, f1, accuracy] = getF1(label_test, p_label)
    print "F1: [%.4f, %.4f, %.4f] Accuracy: %.4f" % (precision, recall, f1, accuracy)
    f_result = open("result/attribute_"+attribute+"_"+side+".csv", "w")
    for i in range(len(p_label)):
        f_result.write(attribute_info[i]+", "+str(int(label_test[i]))+", "+str(int(p_label[i]))+", ")
        f_result.write("[%.4f]\n" % p_val[i][0])
    f_result.close()
    
def test_combined_attribute(attributes, side):
    """
    test accuracy for combined attributes: 
    for an accurate prediction, all combined attributes should be predicted correctly.
    """
    predicts = []
    labels = []
    for attr in attributes:
        resultfile = "result/attribute_" + attr + "_" + side + ".csv"
        if not os.path.isfile(resultfile):
            print "can't open result file: "+resultfile, " (please train your models)\n"
            return
        f_result = open(resultfile, "r")
        lines = f_result.readlines()
        seqframe = []
        predict_of_one_model = []
        label_of_one_model = []
        for item in lines:
            label_start = item.find(", ")
            seqframe.append(item[0:label_start])
            label_end = item.find(",", label_start+1)
            label_of_one_model.append(int(float(item[label_start+1:label_end])))
            predict_start = label_end
            predict_end = item.find(",", predict_start+1)
            predict_of_one_model.append(int(float(item[predict_start+1:predict_end])))
        predicts.append(predict_of_one_model)
        labels.append(label_of_one_model)
    label_array = np.array(labels, np.float32).transpose()
    predict_array = np.array(predicts, np.float32).transpose()
    num_accurate = 0.0
    for r in range(label_array.shape[0]):
        v = 1
        for c in range(label_array.shape[1]):
            if label_array[r,c] == predict_array[r,c]:
                continue
            else:
                v = 0
                break
        num_accurate += v
    print "accuracy for combined attributes:", num_accurate/label_array.shape[0]

def main(argv):
    """
    """
    #prepare feature by detection
#    seqs_train = []
#    seqs_train.append("006")
#    seqs_train.append("007")
#    seqs_train.append("008")
#    seqs_train.append("010")
#    seqs_train.append("012")
#    seqs_train.append("013")
#    seqs_train.append("014")
#    seqs_train.append("016")
#    seqs_train.append("017")
#    seqs_train.append("018")
#    seqs_train.append("021")
#    seqs_train.append("022")
#    seqs_test = []
#    seqs_test.append("002")
#    seqs_test.append("003")
#    seqs_test.append("005")
#    seqs_test.append("020")
#    trainfile = 'data/feature_attribute_train.csv'
#    testfile = 'data/feature_attribute_test.csv'
#    get_feature_by_detection(seqs_train, trainfile)
#    get_feature_by_detection(seqs_test, testfile)
    attributes = []
    attributes.append("flat")
    attributes.append("prismatic")
    attributes.append("rigid")
    attributes.append("sphere")
    for attribute in attributes:
        train_rate, test_rate = get_attribute_freq(attribute, "left")
        print train_rate, test_rate
        train_attribute(attribute, "left")
    test_combined_attribute(attributes, "left")
    for attribute in attributes:
        train_rate, test_rate = get_attribute_freq(attribute, "right")
        print train_rate, test_rate
        train_attribute(attribute, "right")
    test_combined_attribute(attributes, "right")
    
if __name__ == "__main__":
    main(sys.argv[1:])