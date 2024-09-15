# @Author : CyIce
# @Time : 2024/6/27 15:24

import pandas as pd


def read_pathway(path):
    """
    Creating the mapping dict from pathway to related genes list.
    """
    pathways = pd.read_csv(path,sep='\t').values
    pathways_dict = {}
    for line in pathways:
        if line[0] not in pathways_dict.keys():
            pathways_dict[line[0]] = [line[1]]
        else:
            pathways_dict[line[0]].append(line[1])
    return pathways_dict
