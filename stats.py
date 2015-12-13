# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:02:35 2015

@author: osm3000
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def stats_fn():
    data_frame = {}    
    algorithms_list = ["gbrt","adaboost", "hmm", "logistic", "NaiveBayes", "randomforests", "svm"]
    #algorithms_list = ["gbrt","adaboost", "hmm", "logistic", "randomforests", "svm"]
    # Get the stats for each algorthim on its own
    for algo_name in algorithms_list:
        print algo_name
        data_frame[algo_name] = {}
        hyper_vol_file = open("./Results/"+algo_name+".txt", "r")
        hyper_vol_lines = hyper_vol_file.read().splitlines()
        hyper_vol_file.close()
    
        stat_file = open("./stats/Stat_tests_" + algo_name + ".txt", "w")
    
        for line in hyper_vol_lines:
            line_split = line.split(",")
            try:
                data_frame[algo_name]["cv"].append(float(line_split[0]))
            except:
                data_frame[algo_name]["cv"] = [float(line_split[0])]
                
            try:
                data_frame[algo_name]["train"].append(float(line_split[1]))
            except:
                data_frame[algo_name]["train"] = [float(line_split[1])]
                
            try:
                data_frame[algo_name]["test"].append(float(line_split[2]))
            except:
                data_frame[algo_name]["test"] = [float(line_split[2])]
        # This part is for the statistics
        # Here, I will always use my base case as the PAREGO algo_name - Since, from the boxplots - it performed
        # the worst by far.
        statistical_significance = stats.wilcoxon(data_frame[algo_name]["cv"], data_frame[algo_name]["train"])
        print >> stat_file, "cv VS train -->", statistical_significance
        print >> stat_file, "cv median = ", np.median(data_frame[algo_name]["cv"])
        print >> stat_file, "train median = ", np.median(data_frame[algo_name]["train"])
        print >> stat_file, "----------------------------------------------------------"
        
        statistical_significance = stats.wilcoxon(data_frame[algo_name]["cv"], data_frame[algo_name]["test"])
        print >> stat_file, "cv VS test -->", statistical_significance
        print >> stat_file, "cv median = ", np.median(data_frame[algo_name]["cv"])
        print >> stat_file, "test median = ", np.median(data_frame[algo_name]["test"])
        print >> stat_file, "----------------------------------------------------------"
        
        statistical_significance = stats.wilcoxon(data_frame[algo_name]["test"], data_frame[algo_name]["train"])
        print >> stat_file, "test VS train -->", statistical_significance
        print >> stat_file, "test median = ", np.median(data_frame[algo_name]["test"])
        print >> stat_file, "train median = ", np.median(data_frame[algo_name]["train"])
        print >> stat_file, "----------------------------------------------------------"
    
        stat_file.close()
    # Get the stats for between different algorithms
    seen_pairs = []
    stat_file = open("./stats/Stat_tests_ALL.txt", "w")
    for algorithm in data_frame:
        print "algorithm :", algorithm
        for algorithm2 in data_frame:
            if (algorithm != algorithm2) and ((algorithm, algorithm2) not in seen_pairs):
                seen_pairs.append((algorithm, algorithm2))
                seen_pairs.append((algorithm2, algorithm))
                try:
                    statistical_significance = stats.wilcoxon(data_frame[algorithm]["cv"], data_frame[algorithm2]["cv"])
                    print >> stat_file, "Comparing CV results"
                    print >> stat_file, algorithm, " VS ", algorithm2, " -->", statistical_significance
                    print >> stat_file, algorithm, " median = ", np.median(data_frame[algorithm]["cv"])
                    print >> stat_file, algorithm2, " median = ", np.median(data_frame[algorithm2]["cv"])
                    print >> stat_file, "----------------------------------------------------------"
                except:
                    pass
                
                try:
                    statistical_significance = stats.wilcoxon(data_frame[algorithm]["train"], data_frame[algorithm2]["train"])
                    print len(data_frame[algorithm]["train"])
                    print len(data_frame[algorithm2]["train"])
                    print "--------------------------------------------------------------"
                    print >> stat_file, "Comparing Train results"
                    print >> stat_file, algorithm, " VS ", algorithm2, " -->", statistical_significance
                    print >> stat_file, algorithm, " median = ", np.median(data_frame[algorithm]["train"])
                    print >> stat_file, algorithm2, " median = ", np.median(data_frame[algorithm2]["train"])
                    print >> stat_file, "----------------------------------------------------------"
                except:
                    pass
                
                try:
                    statistical_significance = stats.wilcoxon(data_frame[algorithm]["test"], data_frame[algorithm2]["test"])
                    print >> stat_file, "Comparing Test results"
                    print >> stat_file, algorithm, " VS ", algorithm2, " -->", statistical_significance
                    print >> stat_file, algorithm, " median = ", np.median(data_frame[algorithm]["test"])
                    print >> stat_file, algorithm2, " median = ", np.median(data_frame[algorithm2]["test"])
                    print >> stat_file, "----------------------------------------------------------"
                except:
                    pass
        print "********************************************************************"
        print "********************************************************************"
    stat_file.close()
        
stats_fn()