#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:47:27 2018

@author: yahuishi
"""

import pandas as pd
import numpy as np
import re
from collections import OrderedDict

MAX_BULLET = 15    # reasonable largest length of a bullet list

def text_preprocess(inStr):
    outStr = re.sub('<br>|<span>|</span>|</br>', ' ', inStr)
    return outStr
    
def find_numeric_bullets(inStr):
    '''
    Find starting indices of all numeric bullets by regExp matching
    
    Args:
        - inStr (string)
        
    Returns:
        - A dictionary {<index>: <digits found>}
    '''
    bullet_indices = []
    bullet_digit = []
    p_bullet = re.compile('\d+[\.:\)]?\s')
    for m in p_bullet.finditer(inStr):
        bullet_indices.append(m.start())
        matched_str = inStr[m.start():m.end()]
        bullet_digit.append(re.compile('(\d+)[\.:\)]?\s').findall(matched_str)[0])
    return dict(zip(bullet_indices, bullet_digit))

def find_numeric_seq(inStr, idx2dig):
    '''
    Find the longest continuous numeric sequence (ie. 1, 2, 3, ...)
    
    Args:
        - inStr (string): raw sentence
        - idx2dig: return value of function find_numeric_bullets {<index>: <digits found>}
        
    Returns:
        - An ordered dictionary containing continuous numeric sequence {<digit>: <list of indices>}
    '''
    
    num_indice = OrderedDict()
    
    for i in range(1, MAX_BULLET):
        i_indice = [int(idx) for idx, num in idx2dig.items() if int(num) == i]
        if len(i_indice) == 0:
            break
        num_indice[str(i)] = i_indice
    return num_indice
    
def find_possible_path(inStr, dig2idx, n = 1):
    '''
    Find the index sequence split points for the input string through dynamic programming
    
    Args:
        - inStr (string): raw sentence
        - dig2idx: return value of function find_numeric_seq {<digit>: <list of indices>}
        - n (int): number of output solution, default value 1 to select one best solution
        
    Returns:
        - A list, each element is a list of possible split points
    '''
    
    # ========= Dynamic programming =========
    # scoring function to path node1-node2
    def DP_score_func(node_id1, node_id2, dig2idx, inStr):
        if (node_id1 == 'start' and node_id2.startswith('l1e')):
            return 10   # virtual start point -> 1
        if (node_id2 == 'end'):
            return 10    # current point -> virtual end point

        layer_1 = int(re.compile('l(\d+)e').findall(node_id1)[0])
        id_1 = int(re.compile('l\d+e(\d+)').findall(node_id1)[0])
        layer_2 = int(re.compile('l(\d+)e').findall(node_id2)[0])
        id_2 = int(re.compile('l\d+e(\d+)').findall(node_id2)[0])
        if (layer_2 - layer_1 == 1) and (dig2idx[str(layer_2)][int(id_2)] > dig2idx[str(layer_1)][int(id_1)]):
            score = 1    # basic score for increasing index
            suffix_1 = inStr[dig2idx[str(layer_1)][int(id_1)]+len(str(layer_1))]
            suffix_2 = inStr[dig2idx[str(layer_2)][int(id_2)]+len(str(layer_2))]
            if suffix_1 == suffix_2:    # extra bonus if subfix is the same, such as 1. and 2. or 1: and 2:
                score += 9
                # print('bonus for same suffix %s' % suffix_1)
            return score   # positive score for possible edge
        else:
            return -999    # penalty for impossible edge
        
    # DP_scores
    DP_scores = {}    # key: 'l1e2' stands for 2nd element in 1st layer; value: best score for path ends at current position
    DP_parent = {}
    n_layer = len(dig2idx)

    # forward
    end_max_score = -999
    for i_layer in range(1, n_layer+1):
        # print('i_layer: %d' % i_layer)
        n_idx_curLayer = len(dig2idx[str(i_layer)])
        for i_node_curLayer in range(n_idx_curLayer):
            # print('i_node_curLayer: %d' % i_node_curLayer)
            cur_node_id = 'l' + str(i_layer) + 'e' + str(i_node_curLayer)
            if i_layer == 1:    # layer 1
                DP_scores[cur_node_id] = DP_score_func('start', cur_node_id, dig2idx, inStr)
            else:
                n_idx_pre = len(dig2idx[str(i_layer-1)])
                ids_preLayer = ['l' + str(i_layer-1) + 'e' + str(i_node_preLayer) for i_node_preLayer in range(n_idx_pre)]
                step_scores = [(DP_scores[each_id_preL] + DP_score_func(each_id_preL, cur_node_id, dig2idx, inStr)) for each_id_preL in ids_preLayer]
                DP_scores[cur_node_id] = max(step_scores)
                DP_parent[cur_node_id] = ids_preLayer[step_scores.index(max(step_scores))]
            end_score = DP_scores[cur_node_id] + DP_score_func(cur_node_id, 'end', dig2idx, inStr)
            if end_score > end_max_score:
                end_max_score = end_score
                end_parent = cur_node_id
        DP_scores['end_at_l' + str(i_layer)] = end_max_score
        DP_parent['end_at_l' + str(i_layer)] = end_parent
    
    # tracing back
    DP_scores_ordered = sorted(DP_scores.items(), key = lambda it: it[1], reverse = True)
    # find the optimal end points
    if len(DP_scores_ordered) == 0:    # bullets not found
        return None, [0, len(inStr)]

    best_scored_end_layer = int(re.compile('end_at_l(\d+)').findall(DP_scores_ordered[0][0])[0])
    best_end = DP_scores_ordered[0][0]
    if ('end_at_l' + str(best_scored_end_layer-1)) in DP_scores:
        score_stop_at_preL = DP_scores['end_at_l' + str(best_scored_end_layer-1)]
        if DP_scores_ordered[0][1] - score_stop_at_preL < 10:    # abandon the last potential bullet if only index increased, but suffix not match with previous bullet
            best_end = 'end_at_l' + str(best_scored_end_layer-1)

    # construct the path
    path = []
    split_points = []
    cur_node = best_end
    while cur_node in DP_parent:
        parent_node = DP_parent[cur_node]
        path = [parent_node] + path
        cur_node = parent_node
    for node_id in path:
        l, e = re.compile('l(\d+)e(\d+)').findall(node_id)[0]
        split_points.append(dig2idx[l][int(e)])
    return path, split_points
        
def get_split_text(split_points, inStr):
    split_text = []
    if split_points[-1] != len(inStr):
        split_points += [len(inStr)]
    for i_sp in range(len(split_points)-1):
            seg = inStr[split_points[i_sp]:split_points[i_sp+1]]
            split_text.append(seg)
    return split_text

def main(inStr):
    bullets_indices = find_numeric_bullets(inStr)    # find all bullets
    num_seq = find_numeric_seq(inStr, bullets_indices)    # organize found indices by numeric sequence
    path, split_points = find_possible_path(inStr, num_seq)
    split_rlts = get_split_text(split_points, inStr)
    return(split_rlts)

if __name__ == '__main__':
    sampleData = pd.read_csv('sampleData_preprocess.csv')
    
    sampleData.current_condition__c = sampleData.current_condition__c.apply(text_preprocess)    # preprocess
    
    sampleData['current_condition__split'] = sampleData.current_condition__c.apply(main)