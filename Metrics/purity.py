#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:41:12 2017

@author: jorisguerin
"""
import numpy as np
from copy import deepcopy

def confusion_matrix(clusters, classes_gt):
	new_gt = deepcopy(classes_gt)
	l = list(set(classes_gt))
	for i in range(len(classes_gt)):
		for j in range(len(l)):
			if classes_gt[i] == l[j]:
				new_gt[i] = j
				
	conf_mat = np.zeros([len(set(clusters)), len(set(new_gt))])
	for i in range(len(clusters)):
		conf_mat[clusters[i], new_gt[i]] += 1

	return conf_mat

def purity(clusters, classes_gt):
	conf_mat = confusion_matrix(clusters, classes_gt)
	sum_clu  = np.max(conf_mat, axis = 1)
	sum_tot  = np.sum(sum_clu)

	pur = sum_tot / len(clusters)

	return pur

### TEST ###

#classes  = np.array([1,2,0,2,2,1,0,1,0,0,2,2,1,1,1,0])
#clusters = np.array([2,0,1,2,0,1,1,1,2,0,2,2,0,0,0,1])
#
#print(purity(clusters, classes))
#print(purity(classes, clusters))