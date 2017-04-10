# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:23:44 2015

@author: cai-mj
"""

import numpy as np
import os
import sys
import yaml
import grasp_train
from common import read_info, test_eval, transfer_grasp_label
import attribute_train

FLAT_ID = int("00000001", base=2)
PRISMATIC_ID = int("00000010", base=2)
RIGID_ID = int("00000100", base=2)
SPHERE_ID = int("00001000", base=2)

def write_ssvmdata_joint(f_attribute_file, f_grasp_file, dst_file):
    """
    write joint label and feature file for structured svm.
    """
    #read attribute data
    f_attribute = open(f_attribute_file, "r")
    first_line = f_attribute.readline()
    tags = first_line.split(',')
    idx_s = -1
    idx_f = -1
    for i in range(len(tags)):
        if tags[i].find("feature") == 0:
            idx_f = i
        if tags[i] == "side":
            idx_s = i
    assert(idx_f != -1 and idx_s != -1)
    items = f_attribute.readlines()
    f_attribute.close()
    labels_attribute = []
    features_attribute = []   
    info_attribute = []
    for item in items:
        item_split = item.split(',')
        #only process right hand; modify this if double hands are to be process
        if "left" == item_split[idx_s]:
            continue       
        label = [v for v in item_split[idx_s+1:idx_f]]
        labels_attribute.append(label)
        features_attribute.append(item_split[idx_f:len(item_split)-1]) #last item is "\n"
        info_attribute.append("_".join(item_split[0:idx_s+1]))
    #project to PC axis
    data_attribute = np.array(features_attribute, dtype=np.float32)
#    print "attribute data before pca: ", data_attribute.shape
#    data_attribute = attribute_train.feature_pca(data_attribute)
#    print "attribute data after pca: ", data_attribute.shape

    #get frequent grasp types
    grasp_count_train, grasp_count_test = grasp_train.get_freq_grasp(3, 1)
    grasp_freq = []
    sorted_type = sorted(grasp_count_train)
    for item in sorted_type:
        if grasp_count_test.has_key(item):
            grasp_freq.append(item)
    print grasp_freq
    #read grasp data   
    f_grasp = open(f_grasp_file, "r")
    first_line = f_grasp.readline()
    tags = first_line.split(',')
    idx_s = -1
    idx_f = -1
    for i in range(len(tags)):
        if tags[i].find("feature") == 0:
            idx_f = i
        if tags[i] == "side":
            idx_s = i
    assert(idx_f != -1 and idx_s != -1)
    items = f_grasp.readlines()
    f_grasp.close()
    labels_grasp = []
    features_grasp = []   
    info_grasp = []
    for item in items:
        item_split = item.split(',')
        #only process right hand; modify this if double hands are to be process
        if "left" == item_split[idx_s]:
            continue       
        label = [v for v in item_split[idx_s+1:idx_f]]
        labels_grasp.append(label)
        features_grasp.append(item_split[idx_f:len(item_split)-1]) #last item is "\n"
        info_grasp.append("_".join(item_split[0:idx_s+1]))
    #project to PC axis
    data_grasp = np.array(features_grasp, dtype=np.float32)
#    print "grasp data before pca: ", data_grasp.shape
#    data_grasp = grasp_train.feature_pca(data_grasp)
#    print "grasp data after pca: ", data_grasp.shape
    for r in range(len(labels_grasp)):
            for c in range(len(labels_grasp[r])):
                idx = -1
                for g in range(len(grasp_freq)):
                    if labels_grasp[r][c] == grasp_freq[g]:
                        idx = g
                        break
                labels_grasp[r][c] = str(idx)

    #write joint data
    f_joint = open(dst_file, 'w') 
    f_joint.write(str(len(labels_attribute[0]))+","+str(len(labels_grasp[0]))+",")
    f_joint.write(str(data_attribute.shape[1])+","+str(data_grasp.shape[1])+"\n")
    for i in range(len(info_attribute)):
        idx_g = -1
        for j in range(len(info_grasp)):
            if info_attribute[i] == info_grasp[j]:
                idx_g = j
                break
        if idx_g == -1:
            print "WARNING: no matched info for: "+info_attribute[i]
            continue       
                        
        f_joint.write(info_attribute[i]+",")
        f_joint.write(",".join(labels_attribute[i])+","+",".join(labels_grasp[i])+",")
        for j in range(data_attribute.shape[1]):
            f_joint.write(str(data_attribute[i,j])+",")
        for j in range(data_grasp.shape[1]):
            f_joint.write(str(data_grasp[i,j])+",")
        f_joint.write("\n")
    
    
def edge_potential(seqs, attributes, grasps, side):
    """
    get edge potential of object attribute and grasp type from training data
    """
    print "train for edge potential...\n"
    pA = np.zeros(2**len(attributes), np.float32)
    pGbyA = np.zeros((pA.size, len(grasps)), np.float32)
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
            f_yml = open(ymlfile, "r")
            yml_dict = yaml.load(f_yml.read())
            f_yml.close()
            seqname = str(yml_dict["seqname"])
            filename = str(yml_dict["filename"])
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
            is_hLvisible = yml_dict["lefthand"]["visible"]
            hL_grasp = yml_dict["lefthand"]["grasp"]
            item_split = hL_grasp.split()
            hL_grasp = "-".join(item_split)
            hL_grasp = transfer_grasp_label(hL_grasp)
            is_hRvisible = yml_dict["righthand"]["visible"]
            hR_grasp = yml_dict["righthand"]["grasp"]
            item_split = hR_grasp.split()
            hR_grasp = "-".join(item_split)
            hR_grasp = transfer_grasp_label(hR_grasp)
            if is_oLvisible == 1 and is_hLvisible == 1 and side == "left":
                idx_attribute = 0
                if is_oLflat==1:
                    idx_attribute = idx_attribute | FLAT_ID
                if is_oLprismatic==1:
                    idx_attribute = idx_attribute | PRISMATIC_ID
                if is_oLrigid==1:
                    idx_attribute = idx_attribute | RIGID_ID
                if is_oLsphere==1:
                    idx_attribute = idx_attribute | SPHERE_ID
                pA[idx_attribute] += 1 
                idx_grasp = -1
                for i in range(len(grasps)):
                    if hL_grasp == grasps[i]:
                        idx_grasp = i
                if idx_grasp != -1:
                    pGbyA[idx_attribute, idx_grasp] += 1
            if is_oRvisible == 1 and is_hRvisible == 1 and side == "right":
                idx_attribute = 0
                if is_oRflat==1:
                    idx_attribute = idx_attribute | FLAT_ID
                if is_oRprismatic==1:
                    idx_attribute = idx_attribute | PRISMATIC_ID
                if is_oRrigid==1:
                    idx_attribute = idx_attribute | RIGID_ID
                if is_oRsphere==1:
                    idx_attribute = idx_attribute | SPHERE_ID
                pA[idx_attribute] += 1
                idx_grasp = -1
                for i in range(len(grasps)):
                    if hR_grasp == grasps[i]:
                        idx_grasp = i
                if idx_grasp != -1:
                    pGbyA[idx_attribute, idx_grasp] += 1
    pA.__idiv__(np.sum(pA)) #divide by sum to get proportion
    pGA = np.zeros(pGbyA.shape, pGbyA.dtype)
    L = 1.0 #universally added number
    pGbyA.__iadd__(L)
    for i in range(pGbyA.shape[0]):
        if np.sum(pGbyA[i,:]) == 0:
            continue
        pGbyA[i,:].__idiv__(np.sum(pGbyA[i,:])) #divide by line sum to get proportion
        for j in range(pGbyA.shape[1]):
            pGA[i,j] = pGbyA[i,j] #* pA[i] uncomment when consider different prior
    return pGA
    

def argmax_potential(a_score, g_score, pGA):
    """
    search best combination of object attribute and grasp type with maximum potential\n
    a_score: node potential of object attribute\n
    g_score: node potential of grasp type\n
    pGA: trained edge potential
    """
    assert (pGA.shape[0] == 2**a_score.size and pGA.shape[1] == g_score.size)
    max_potential = 0
    max_idx_a = -1
    max_idx_g = -1
    for idx_a in range(2**a_score.size):
        for idx_g in range(g_score.size):
            a_potential = 1
            for i in range(a_score.size):
                if 2**i & idx_a > 0:
                    a_potential *= a_score[i]
                else:
                    a_potential *= 1-a_score[i]
            g_potential = g_score[idx_g]
            potential = a_potential*g_potential*pGA[idx_a,idx_g]
            if potential > max_potential:
                max_idx_a = idx_a
                max_idx_g = idx_g
                max_potential = potential
    assert(max_potential > 0)
    g_predict = max_idx_g
    a_predict = []
    for i in range(a_score.size):
        if 2**i & max_idx_a > 0:
            a_predict.append(1)
        else:
            a_predict.append(0)
    return [a_predict, g_predict]
    
    
def joint_inference(attributes, grasps, pGA, side):
    """
    joint inference of object attributes and grasp types\n
    pGA: array of edge potential between grasp and object attribute
    """
    attribute_info = read_info("data/feature_attribute_test.csv", side)
    grasp_info = read_info("data/feature_grasp_test.csv", side)
    #read decision probability of each grasp type
    g_scores = []
    g_labels = []
    for grasp_type in grasps:
        resultfile = "result/grasp_" + grasp_type + "_" + side + ".csv"
        if not os.path.isfile(resultfile):
            print "can't open result file: "+resultfile, " (please train your models)\n"
            return
        f_result = open(resultfile, "r")
        lines = f_result.readlines()
        score_of_one_model = []
        label_of_one_model = []
        for item in lines:
            label_start = item.find(", ")
            label_end = item.find(",", label_start+1)
            label_of_one_model.append(int(float(item[label_start+1:label_end])))
            score_start = item.find("[")
            score_end = item.find("]", score_start+1)
            score_of_one_model.append(float(item[score_start+1:score_end]))
        g_scores.append(score_of_one_model)
        g_labels.append(label_of_one_model)
    g_score_array = np.array(g_scores, np.float32).transpose()
    g_label_array = np.array(g_labels, np.int32).transpose()
    #read decision probability of each binary attribute
    a_scores = []
    a_labels = []
    for attribute in attributes:
        resultfile = "result/attribute_" + attribute + "_" + side + ".csv"
        if not os.path.isfile(resultfile):
            print "can't open result file: "+resultfile, " (please train your models)\n"
            return
        f_result = open(resultfile, "r")
        lines = f_result.readlines()
        score_of_one_model = []
        label_of_one_model = []
        for item in lines:
            label_start = item.find(", ")
            label_end = item.find(",", label_start+1)
            label_of_one_model.append(int(float(item[label_start+1:label_end])))
            score_start = item.find("[")
            score_end = item.find("]", score_start+1)
            score_of_one_model.append(float(item[score_start+1:score_end]))
        a_scores.append(score_of_one_model)
        a_labels.append(label_of_one_model)
    a_score_array = np.array(a_scores, np.float32).transpose()
    a_label_array = np.array(a_labels, np.int32).transpose()

    #joint inference
    label_gt_g = []
    label_gt_a = []
    label_predict_a = []
    label_predict_g = []
    f_log = open("result/join_"+side+".csv", "w")
    for i in range(len(attribute_info)):
        idx_g = -1
#        print "joint infer: "+attribute_info[i]
        for j in range(len(grasp_info)):
            if attribute_info[i] == grasp_info[j]:
                idx_g = j
                break
        if idx_g == -1:
            print "WARNING: no matched info for: "+attribute_info[i]
            continue
        temp_g = np.argmax(g_label_array[idx_g,:])
        if g_label_array[idx_g, temp_g] < 1:
            temp_g = -1 #for some case when the label doesnot belong to freq labels
            continue
        label_gt_g.append(temp_g)
        temp_a = a_label_array[i,:].tolist()
        label_gt_a.append(temp_a)
        a_predict, g_predict = argmax_potential(a_score_array[i,:],g_score_array[idx_g,:], pGA)
        label_predict_g.append(g_predict)
        label_predict_a.append(a_predict)        
        f_log.write(attribute_info[i]+", ")
        for j in range(len(temp_a)):
            f_log.write(str(temp_a[j]))
        f_log.write(", ")
        for j in range(a_score_array.shape[1]):
            f_log.write(str(int(a_score_array[i,j]+0.5)))
        f_log.write("->")
        for j in range(len(a_predict)):
            f_log.write(str(a_predict[j]))
        f_log.write(", "+str(temp_g))
        f_log.write(", "+str(np.argmax(g_score_array[idx_g,:]))+"->"+str(g_predict)+"\n")

    label_predict_a = np.array(label_predict_a, np.int32)
    label_predict_g = np.array(label_predict_g, np.int32)
    label_gt_a = np.array(label_gt_a, np.int32)
    label_gt_g = np.array(label_gt_g, np.int32)
    for i in range(len(attributes)):
        [precision, recall, f1, accuracy] = attribute_train.getF1(label_gt_a[:,i], label_predict_a[:,i])
        print attributes[i], "F1: [%.4f, %.4f, %.4f] Accuracy: %.4f" % (precision, recall, f1, accuracy)
    num_accurate = 0.0
    for r in range(label_gt_a.shape[0]):
        v = 1
        for c in range(label_gt_a.shape[1]):
            if label_gt_a[r,c] == label_predict_a[r,c]:
                continue
            else:
                v = 0
                break
        num_accurate += v
    print "accuracy for combined attributes:", num_accurate/label_gt_a.shape[0]
    test_eval(label_gt_g, label_predict_g, grasps)


def joint(seqs_train, attributes, side):
    """
    """
    grasp_count_train, grasp_count_test = grasp_train.get_freq_grasp(side, 3, 1)
    grasp_freq = []
    sorted_type = sorted(grasp_count_train)
    for item in sorted_type:
        if grasp_count_test.has_key(item):
            grasp_freq.append(item)
    
    pGA = np.zeros((0,0))
    edge_file = "result/prior_"+side+".npy"
    if not os.path.isfile(edge_file):
        pGA = edge_potential(seqs_train, attributes, grasp_freq, side)
        np.save(edge_file, pGA)
        f_edge = open("result/prior_"+side+".csv", "w")
        for i in range(pGA.shape[0]):
            for j in range(pGA.shape[1]):
                f_edge.write("%.4f, " % pGA[i,j])
            f_edge.write("\n")
        f_edge.close
    else:
        pGA = np.load(edge_file)
    joint_inference(attributes, grasp_freq, pGA, side)    
    
    
def main(argv):
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
    
    seqs_train.append("002")
    seqs_train.append("003")
    seqs_train.append("005")
    seqs_train.append("020")
    
    attributes = []
    attributes.append("flat")
    attributes.append("prismatic")
    attributes.append("rigid")
    attributes.append("sphere")
    joint(seqs_train, attributes, "left")
    joint(seqs_train, attributes, "right")
    print "finished"


def run(argv):
    """
    """
    f_attribute_file = "data/feature_attribute_train.csv"
    f_grasp_file = "data/feature_grasp_train.csv"
    dst_file = "data/feature_joint_train.csv"
    write_ssvmdata_joint(f_attribute_file, f_grasp_file, dst_file)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])