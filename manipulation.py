# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:18:44 2015

@author: cai-mj
"""

import numpy as np
import os
import sys
import math
import yaml
import svmutil
import cv2
from skimage.feature import hog
from skimage import color
import caffe
from common import test_eval, read_model_linearSVM
import grasp_train
import attribute_train

def read_info(filename):
    """
    read header info from feature.csv\n
    """
    f_feat = open(filename, "r")
    features = f_feat.readlines()
    f_feat.close()
    info = []
    for item in features:
        item_split = item.split(',')
        info.append(item_split[0])
    return info


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
    
    
def get_manipulation_count(seqs, limit):
    """
    """
    verb_count = {}
    for seq in seqs:
        annotate_filename = "/home/cai-mj/_GTA/annotation/GTEA/"+seq+".txt"
        f_annotate = open(annotate_filename, 'r')
        lines = f_annotate.readlines()
        for line in lines:
            p_start = line.find("<") + 1
            p_end = line.find(">", p_start)
            if p_start == -1 or p_end == -1:
                continue
            verb = line[p_start:p_end]
            if verb_count.has_key(verb):
                verb_count[verb] = verb_count[verb] + 1
            else:
                verb_count[verb] = 1
        f_annotate.close()
    sorted_item = sorted(verb_count)
    for item in sorted_item:
        print item, verb_count[item]
        if verb_count[item] < limit:
            del verb_count[item]        
    return verb_count
    
def isTwoHandExist(filename):
    """
    check in the annotation file whether two hands are both annotated
    """
    f_yml = open(filename, "r")
    yml_dict = yaml.load(f_yml.read())
    f_yml.close()
    is_hLvisible = yml_dict["lefthand"]["visible"]
    is_hRvisible = yml_dict["righthand"]["visible"]
    if is_hLvisible*is_hRvisible == 1:
        return True
    else:
        return False

def get_manipulation_instance(seqs, limit):
    """
    """
    annotateDir = "/home/cai-mj/_GTA/annotation/GTEA_plus/"
    imgDir = "/home/cai-mj/_GTA/img/GTEA_plus/"
    dstDir = "manipulate/data_test/"
    verb_count = {}
    for seq in seqs:
        annotate_filename = annotateDir+seq+".txt"
        f_annotate = open(annotate_filename, 'r')
        lines = f_annotate.readlines()
        f_annotate.close()
        for line in lines:
            p_start = line.find("<") + 1
            p_end = line.find(">", p_start)
            if p_start == -1 or p_end == -1:
                continue
            verb = line[p_start:p_end]
            p_start = line.find("(", p_end) + 1
            p_end = line.find(")", p_start)
            if p_start == -1 or p_end == -1:
                continue
            [frame_start, frame_end] = [int(v) for v in line[p_start:p_end].split("-")]
            for i in range(frame_start, frame_end+1, 1):
                frameid = "%08d" % i
                ymlfile = seq+"/"+frameid+".yml"
                if os.path.isfile(ymlfile) and isTwoHandExist(ymlfile):
                    if verb_count.has_key(verb):
                        verb_count[verb].append(seq+"_"+frameid)
                    else:
                        verb_count[verb] = [seq+"_"+frameid]
                        os.system("mkdir "+dstDir+verb)
                    os.system("cp "+ymlfile+" "+dstDir+verb+"/"+seq+"_"+frameid+".yml")
                    os.system("cp "+imgDir+seq+"/"+frameid+".jpg"+" "+dstDir+verb+"/"+seq+"_"+frameid+".jpg")
    sorted_item = sorted(verb_count)
    for item in sorted_item:
        print item, len(verb_count[item])
#        if len(verb_count[item]) < limit:
#            del verb_count[item]
    return verb_count


def load_pretrained_models():
    """
    """
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
    attributes = []
    attributes.append("flat")
    attributes.append("prismatic")
    attributes.append("rigid")
    attributes.append("sphere")
    #left grasp model
    grasp_count_train, grasp_count_test = grasp_train.get_freq_grasp("left", 3, 1)
    grasp_freq_left = []
    sorted_type = sorted(grasp_count_train)
    for item in sorted_type:
        if grasp_count_test.has_key(item):
            grasp_freq_left.append(item)
    m_grasp_l = []
    for v in grasp_freq_left:
        modelfile = "model/model_grasp_"+v+"_left"
        m = svmutil.svm_load_model(modelfile)
        m_grasp_l.append(m)
    #right grasp model
    grasp_count_train, grasp_count_test = grasp_train.get_freq_grasp("right", 3, 1)
    grasp_freq_right = []
    sorted_type = sorted(grasp_count_train)
    for item in sorted_type:
        if grasp_count_test.has_key(item):
            grasp_freq_right.append(item)
    m_grasp_r = []
    for v in grasp_freq_right:
        modelfile = "model/model_grasp_"+v+"_right"
        m = svmutil.svm_load_model(modelfile)
        m_grasp_r.append(m)
    #left attribute model
    m_attribute_l = []
    for v in attributes:
        modelfile = "model/model_attribute_"+v+"_left"
        m = svmutil.svm_load_model(modelfile)
        m_attribute_l.append(m)
    #right attribute model
    m_attribute_r = []
    for v in attributes:
        modelfile = "model/model_attribute_"+v+"_right"
        m = svmutil.svm_load_model(modelfile)
        m_attribute_r.append(m)
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
    
    return [m_grasp_l, m_grasp_r, m_attribute_l, m_attribute_r, m_target_l, m_target_r]
    

def prepare_mid_features(mnp_freq):
    """
    """
    [m_grasp_l, m_grasp_r, m_attribute_l, m_attribute_r, m_target_l, m_target_r] = load_pretrained_models()
    #get pretrained network
    LAYER = 'pool5'
    LAYER_T = 'pool5'
    INDEX = 4    
    net = get_deep_net()
    img_dir = "/home/cai-mj/_GTA/img/GTEA"
    f_traindata = open("manipulate/data/feature_mnp_train.csv", 'w')
    f_testdata = open("manipulate/data/feature_mnp_test.csv", 'w')
    for mnp in mnp_freq:
        print("extract feature for: "+mnp)
        command = "ls manipulate/data/" + mnp + "/*.yml > filename.txt"
        os.system(command)
        f_files = open("filename.txt", "r")
        files = f_files.readlines()
        f_files.close()  
        feats_gl = []
        feats_tl = []
        feats_gr = []
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
            #hog
            img = cv2.imread(img_dir+"/"+seqname+"/"+filename+".jpg")
            imgroi = img[hL_ymin:hL_ymax+1, hL_xmin:hL_xmax+1]
            imgroi = color.rgb2gray(imgroi)
            imgroi = cv2.resize(imgroi, (80,80))
            feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            feat_gl = feat_hog.tolist()
            imgroi = img[hR_ymin:hR_ymax+1, hR_xmin:hR_xmax+1]
            imgroi = color.rgb2gray(imgroi)
            imgroi = cv2.resize(imgroi, (80,80))
            feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            feat_gr = feat_hog.tolist()
            #cnn
            img = caffe.io.load_image(img_dir+"/"+seqname+"/"+filename+".jpg")
            assert(is_hLvisible == 1 and is_hRvisible == 1)
            imgroi = img[hL_ymin:hL_ymax+1, hL_xmin:hL_xmax+1]
            net.predict([imgroi])
            #feat_gl = net.blobs[LAYER].data[INDEX].flatten().tolist()            
            feats_gl.append(feat_gl)
            feat_tl = net.blobs[LAYER_T].data[INDEX].flatten().tolist()
            feats_tl.append(feat_tl)
            boxes_hl.append([hL_xmin,hL_ymin,hL_xmax,hL_ymax])
            imgroi = img[hR_ymin:hR_ymax+1, hR_xmin:hR_xmax+1]
            net.predict([imgroi])
            #feat_gr = net.blobs[LAYER].data[INDEX].flatten().tolist()            
            feats_gr.append(feat_gr)
            feat_tr = net.blobs[LAYER_T].data[INDEX].flatten().tolist()
            feats_tr.append(feat_tr)
            boxes_hr.append([hR_xmin,hR_ymin,hR_xmax,hR_ymax])
        print("  predict object bounding box...")
        label = [0]*len(files)
        rX_left, p_acc, p_val = svmutil.svm_predict(label, feats_tl, m_target_l[0], '-b 0')
        rY_left, p_acc, p_val = svmutil.svm_predict(label, feats_tl, m_target_l[1], '-b 0')
        rS_left, p_acc, p_val = svmutil.svm_predict(label, feats_tl, m_target_l[2], '-b 0')
        rX_right, p_acc, p_val = svmutil.svm_predict(label, feats_tr, m_target_r[0], '-b 0')
        rY_right, p_acc, p_val = svmutil.svm_predict(label, feats_tr, m_target_r[1], '-b 0')
        rS_right, p_acc, p_val = svmutil.svm_predict(label, feats_tr, m_target_r[2], '-b 0')
        feats_ol = []
        feats_or = []
        print("  object cnn feature...")
        for j in range(len(files)):
            [hL_xmin,hL_ymin,hL_xmax,hL_ymax] = boxes_hl[j]
            pX_left = rX_left[j]*(hL_xmax-hL_xmin+1) + (hL_xmax+hL_xmin)/2
            pY_left = rY_left[j]*(hL_ymax-hL_ymin+1) + (hL_ymax+hL_ymin)/2
            pS_left = rS_left[j]*math.sqrt((hL_xmax-hL_xmin+1)*(hL_ymax-hL_ymin+1))
            [oL_xmin,oL_ymin,oL_xmax,oL_ymax] = [pX_left-pS_left/2,pY_left-pS_left/2,pX_left+pS_left/2,pY_left+pS_left/2]
            [hR_xmin,hR_ymin,hR_xmax,hR_ymax] = boxes_hr[j]
            pX_right = rX_right[j]*(hR_xmax-hR_xmin+1) + (hR_xmax+hR_xmin)/2
            pY_right = rY_right[j]*(hR_ymax-hR_ymin+1) + (hR_ymax+hR_ymin)/2
            pS_right = rS_right[j]*math.sqrt((hR_xmax-hR_xmin+1)*(hR_ymax-hR_ymin+1))
            [oR_xmin,oR_ymin,oR_xmax,oR_ymax] = [pX_right-pS_right/2,pY_right-pS_right/2,pX_right+pS_right/2,pY_right+pS_right/2]
            #read from each yaml file
            ymlfile = files[j][:len(files[j])-1]
            f_yml = open(ymlfile, "r")
            yml_dict = yaml.load(f_yml.read())
            f_yml.close()
            seqname = str(yml_dict["seqname"])
            filename = str(yml_dict["filename"])
            #hog
            img = cv2.imread(img_dir+"/"+seqname+"/"+filename+".jpg")
            imgroi = img[oL_ymin:oL_ymax+1, oL_xmin:oL_xmax+1]
            imgroi = color.rgb2gray(imgroi)
            imgroi = cv2.resize(imgroi, (80,80))
            feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            feat_ol = feat_hog.tolist()
            imgroi = img[oR_ymin:oR_ymax+1, oR_xmin:oR_xmax+1]
            imgroi = color.rgb2gray(imgroi)
            imgroi = cv2.resize(imgroi, (80,80))
            feat_hog = hog(imgroi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            feat_or = feat_hog.tolist()
            #cnn
            img = caffe.io.load_image(img_dir+"/"+seqname+"/"+filename+".jpg")            
            imgroi = img[oL_ymin:oL_ymax+1, oL_xmin:oL_xmax+1]
            net.predict([imgroi])
            #feat_ol = net.blobs[LAYER].data[INDEX].flatten().tolist()            
            feats_ol.append(feat_ol)            
            imgroi = img[oR_ymin:oR_ymax+1, oR_xmin:oR_xmax+1]
            net.predict([imgroi])
            #feat_or = net.blobs[LAYER].data[INDEX].flatten().tolist()
            feats_or.append(feat_or)
        print("  accumulate scores as mid features...")
        scores = []
#        for j in range(len(m_grasp_l)):
#            [p_label, p_acc, p_val] = svmutil.svm_predict(label, feats_gl, m_grasp_l[j], '-b 1')
#            scores.append([p_val[k][0] for k in range(len(p_val))])
#        for j in range(len(m_grasp_r)):
#            [p_label, p_acc, p_val] = svmutil.svm_predict(label, feats_gr, m_grasp_r[j], '-b 1')
#            scores.append([p_val[k][0] for k in range(len(p_val))])
#        for j in range(len(m_attribute_l)):
#            [p_label, p_acc, p_val] = svmutil.svm_predict(label, feats_ol, m_attribute_l[j], '-b 1')
#            scores.append([p_val[k][0] for k in range(len(p_val))])
#        for j in range(len(m_attribute_r)):
#            [p_label, p_acc, p_val] = svmutil.svm_predict(label, feats_or, m_attribute_r[j], '-b 1')
#            scores.append([p_val[k][0] for k in range(len(p_val))])
#        score_array = np.array(scores, np.float32).transpose()
#        for r in range(score_array.shape[0]):
#            gl_sum = np.sum(score_array[r,0:len(m_grasp_l)])
#            for c in range(0,len(m_grasp_l),1):
#                score_array[r,c] = score_array[r,c]/(gl_sum+0.0001)
#            gr_sum = np.sum(score_array[r,len(m_grasp_l):len(m_grasp_l)+len(m_grasp_r)])
#            for c in range(len(m_grasp_l),len(m_grasp_l)+len(m_grasp_r),1):
#                score_array[r,c] = score_array[r,c]/(gr_sum+0.0001)
        for j in range(len(feats_gl)):
            score = []
            score = score + feats_gl[j]
            score = score + feats_gr[j]
            score = score + feats_ol[j]
            score = score + feats_or[j]
            scores.append(score)
        score_array = np.array(scores, np.float32)
#        score_gl = np.array(score_array[:,0:len(feats_gl[0])], score_array.dtype)
#        score_gl = grasp_train.feature_pca(score_gl)
#        score_gr = np.array(score_array[:,len(feats_gl[0]):len(feats_gl[0])+len(feats_gr[0])], score_array.dtype)
#        score_gr = grasp_train.feature_pca(score_gr)
#        score_ol = np.array(score_array[:,len(feats_gl[0])+len(feats_gr[0]):len(feats_gl[0])+len(feats_gr[0])+len(feats_ol[0])], score_array.dtype)
#        score_ol = attribute_train.feature_pca(score_ol)
#        score_or = np.array(score_array[:,len(feats_gl[0])+len(feats_gr[0])+len(feats_ol[0]):len(feats_gl[0])+len(feats_gr[0])+len(feats_ol[0])+len(feats_or[0])], score_array.dtype)
#        score_or = attribute_train.feature_pca(score_or)
#        score_array = np.array(score_gl.transpose())
#        score_array = np.append(score_array, score_gr.transpose(), axis=0)
#        score_array = np.append(score_array, score_ol.transpose(), axis=0)
#        score_array = np.append(score_array, score_or.transpose(), axis=0)
#        score_array = score_array.transpose()
        print "score size:", score_array.shape
        assert(score_array.shape[0] == len(files))
        #save mid features
        for j in range(0, int(len(files)*4/5), 1):
            p_start = files[j].find(mnp+"/")+len(mnp+"/")
            p_end = files[j].find(".yml")
            seqframe = files[j][p_start:p_end]
            f_traindata.write(seqframe+","+mnp+",")
            for k in range(score_array.shape[1]):
                f_traindata.write(str(score_array[j,k])+",")
            f_traindata.write("\n")
        for j in range(0, len(files), 1):
            p_start = files[j].find(mnp+"/")+len(mnp+"/")
            p_end = files[j].find(".yml")
            seqframe = files[j][p_start:p_end]
            f_testdata.write(seqframe+","+mnp+",")
            for k in range(score_array.shape[1]):
                f_testdata.write(str(score_array[j,k])+",")
            f_testdata.write("\n")
    f_traindata.close()
    f_testdata.close()
            
    

def write_svmdata_mnp(srcfile, datafile, mnp_type, isTest):
    """
    """
    f_feat = open(srcfile, 'r')
    lines = f_feat.readlines()
    labels = []
    features = []
    for i in range(len(lines)):
        line = lines[i]
        item_split = line.split(",")
        #right now, the formant is seqframe,mnp_type,feature...
        label_mnp = item_split[1]
        label = 0
        if label_mnp == mnp_type:
            label = 1
        labels.append(label)
        feature = [float(v) for v in item_split[2:len(item_split)-1]]        
        features.append(feature[0:len(feature)]) #grasp type + object attribute
#        features.append(feature[0:17]) # grasp type alone
#        features.append(feature[17:25]) # object attribute alone
    data = np.array(features, dtype=np.float32)
    print "data size:", data.shape
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
    

def train_manipulation(mnp_type):
    """  
    """
    #train
    datafile = "manipulate/model/traindata_mnp_"+mnp_type
    if not os.path.isfile(datafile):
        srcfile = "manipulate/data/feature_mnp_train.csv"
        write_svmdata_mnp(srcfile, datafile, mnp_type, 0)    
    label_train,data_train = svmutil.svm_read_problem(datafile)
    modelfile = "manipulate/model/model_mnp_"+mnp_type
    m = []
    if not os.path.isfile(modelfile):
        print("train model: " + mnp_type)
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
        m = svmutil.svm_train(prob, param)        
        svmutil.svm_save_model(modelfile, m)
    else:
        print("load model: " + mnp_type)
        m = svmutil.svm_load_model(modelfile)
#    weight = read_model_linearSVM(modelfile, len(data_train[0]))
#    print weight
    #test    
    mnp_info = read_info("manipulate/data/feature_mnp_test.csv")
    datafile = "manipulate/model/testdata_mnp_"+mnp_type
    if not os.path.isfile(datafile):
        srcfile = "manipulate/data/feature_mnp_test.csv"
        write_svmdata_mnp(srcfile, datafile, mnp_type, 1)    
    label_test,data_test = svmutil.svm_read_problem(datafile)
    p_label, p_acc, p_val = svmutil.svm_predict(label_test, data_test, m, '-b 1')
    f_result = open("manipulate/result/mnp_" + mnp_type + ".csv", "w")
    for i in range(len(p_label)):
        f_result.write(mnp_info[i]+", "+str(int(label_test[i]))+", "+str(int(p_label[i]))+", ")
        f_result.write("[%.4f]\n" % p_val[i][0])
    f_result.close()


def test_multi_mnp(mnp_freq):
    """
    predict manipulation type with highest score from multiple models
    """
    mnp_info = []
    scores = []
    labels = []
    for mnp_type in mnp_freq:
        resultfile = "manipulate/result/mnp_" + mnp_type + ".csv"
        if not os.path.isfile(resultfile):
            print "can't open result file: "+resultfile, " (please train your models)\n"
            return
        f_result = open(resultfile, "r")
        lines = f_result.readlines()
        seqframe = []
        score_of_one_model = []
        label_of_one_model = []
        for item in lines:
            label_start = item.find(",")
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
        del mnp_info[del_index[i]]
    test_eval(label_gt, label_predict, mnp_freq)  
    f_result = open("manipulate/result/mnp_all.csv", "w")
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
    seqs_train = []
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
#    seqs_train.append("002")
#    seqs_train.append("003")
#    seqs_train.append("005")
#    seqs_train.append("020")
    seqs_train.append("Alireza_American")
    seqs_train.append("Alireza_Snack")
#    mnp_count_train = get_manipulation_instance(seqs_train, 10)
#    mnp_freq = sorted(mnp_count_train)
    mnps = []
    mnps.append("close")
    mnps.append("cut")
    mnps.append("open")
    mnps.append("pour")
    mnps.append("scoop")
    mnps.append("spread")
    mnps.append("stack")
    
#    mnps.append("put")
#    mnps.append("take")
#    prepare_mid_features(mnps)
    for mnp_type in mnps:
        train_manipulation(mnp_type)
    test_multi_mnp(mnps)
    print "finished"
    
    
if __name__ == "__main__":
    main(sys.argv[1:])