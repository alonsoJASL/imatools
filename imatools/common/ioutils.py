import glob
import os
import platform as pltf
import sys
import logging

import numpy as np


def l2_norm(a): return np.linalg.norm(a, axis=1)
def dot_prod_vec(a,b): return np.sum(a*b, axis=1)

def ext(fname, extension) : 
    """
    Returns filename with extension
    """
    xt = f'.{extension}' if extension[0] != '.' else extension
    fname = fname if fname[-len(xt):] == xt else fname + xt
    return fname

def get_subfolders(directory: str) -> list:
    """
    Returns list of subfolders in a directory
    """
    return [f.path for f in os.scandir(directory) if f.is_dir()]

def find_file(directory: str, fname: str, extension='') -> str:
    """
    Returns path of file in a directory
    """
    list_of_files = []
    for name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, name)):
            if fname in name :
                list_of_files.append(os.path.join(directory, name))

    if len(list_of_files) == 0 :
        return ''

    if len(list_of_files) == 1 :
        return list_of_files[0]

    if len(list_of_files) > 1 :
        if extension == '' :
            return list_of_files[0]
        else :
            for f in list_of_files :
                if extension in f :
                    return f 

    return ''

def slot_in_path_hrchy(filepath: str, fname='',  num_levels_above=1) -> str:
    """
    Returns path of file in a directory
    """
    filepath = os.path.normpath(filepath)
    num_levels_above = np.abs(num_levels_above)

    res = '/'.join(filepath.split('/')[0:-num_levels_above])
    if fname != '' :
        res = os.path.join(res, fname)
    
    return res

def fullfile(*paths):
    """
    Returns path separated by '/'
    """
    s='/'
    return s.join(paths)

def mkdirplus(*paths) : 
    """
    Joins paths with fullfile, then creates path 
    returns path
    """
    res = fullfile(*paths)
    os.makedirs(res, exist_ok=True)
    return res 


def searchFileByType(directory, prefix='', extension=''):
    """
    Search file by filetype
    """
    l = glob.glob(fullfile(directory, prefix + '*.' + extension))
    return l

def cout(msg, typeMsg="INFO", print2console=True, logger=None):
    """
    LOGGING FUNCTION
    """
    if print2console==True:
        if logger is not None:
            
            logger.info(f"_ {msg}")
        else:
            print(f"[{typeMsg}] {msg}")

def getTotal(fname):
    """
    Get total number of elements at the top of a file
    """
    try:
        with open(fname, encoding='utf-8') as f:
            numNodes=int(f.readline().strip())

    except Exception as e:
        print("[getTotal] Error - file not found")
        sys.exit(-1)

    return numNodes

def getFileContentWithTotal(fname):
    """
    Get first line separated from the rest (as long string)
    """
    try:
        with open(fname, encoding='utf-8') as f:
            numNodes=int(f.readline().strip())
            restOfFile=f.readlines()

    except Exception as e:
        print("[getTotal] Error - file not found")
        sys.exit(-1)

    return numNodes, restOfFile

def check_file(file):
    if not os.path.isfile(file):
        raise Exception(f"With the options selected, you need to have {file}")

def read_pts(filename):
    print(f'Reading: {filename}')
    return np.loadtxt(filename, dtype=float, skiprows=1)

ELEM_TYPES = ['Tt', 'Tr', 'Ln']
def read_elem(filename, el_type='Tt', tags=True):
    if el_type not in ELEM_TYPES:
        raise Exception('element type not recognised. Accepted: Tt, Tr, Ln')
        
    cols_notags_dic = {'Tt':(1,2,3,4),'Tr':(1,2,3),'Ln':(1,2)}
    cols = cols_notags_dic[el_type]
    if tags:
        # add tags column (largest + 1)
        cols += (cols[-1]+1,)
        
    return np.loadtxt(filename, dtype=int, skiprows=1, usecols=cols)

def read_lon(filename):
    print(f'Reading: {filename}')
    return np.loadtxt(filename, dtype=float, skiprows=1)

def readParsePts(ptsFname):
    """
    Read parse CARP point files
    """
    numNodes=getTotal(ptsFname)
    nodes=read_pts(ptsFname)

    if (numNodes != len(nodes)):
        print("Error in file")
        raise Exception("Error in file")

    return nodes, numNodes

    
def readParseElem(elFname):
    """
    Read and parse CARP element file
    """
    nElem = getTotal(elFname)
    el = read_elem(elFname)

    if (nElem != len(el)):
        print("Error in file")
        raise Exception("Error in file")
    
    return el, nElem

def readFileToList(fname, delim=','):
    """
    Read File to list. Input is normally a table, like a csv
    """
    try:
        with open(fname, encoding='utf-8') as f:
            fileContents=f.readlines()
            fileContentsInList = [(line.strip()).split(sep=delim) for line in fileContents]
            return fileContentsInList

    except Exception as e:
        print("[readFileToList] Error - file not found")
        sys.exit(-1)

def loadCarpMesh(mshname, directory=None):
    """
    Load CARP mesh. Supports for triangle (Tr) and tetrahedral (Tt) meshes
    """

    if directory is not None:
        ptsname = fullfile(directory, mshname+'.pts')
        elemname = fullfile(directory, mshname + ".elem")
    else:
        ptsname = mshname+'.pts'
        elemname = mshname + ".elem"

    pts, nPts = readParsePts(ptsname)
    el, nElem = readParseElem(elemname)

    elem=list()
    for e in el:
        nel = 4 if e[0]=='Tr' else 5
        elem_before = e[1:nel]
        elem.append([int(ex.strip()) for ex in elem_before])

    region_before = [e[-1] for e in el]
    region = [int(x.strip()) for x in region_before]

    return pts, elem, np.asarray(region, dtype=int)


def saveToCarpTxt(pts, el, mshname):
    np.savetxt(mshname+'.pts', pts, header=str(len(pts)), comments='', fmt='%6.12f')
    np.savetxt(mshname+'.elem', el, header=str(len(el)), comments='', fmt='Tr %d %d %d 1')

def near(value1, value2, tol=1e-8) : 
    return (np.abs(value1-value2) <= tol)

def chooseplatform() : 
    return pltf.platform().split('-')[0]

def performanceMetrics(tp, tn, fp, fn) : 

    den_jaccard = tp+fn+fp
    den_precision = tp+fp
    den_recall = tp+fn
    den_accuracy = tp+tn+fp+fn
    den_dice = 2*tp + fp + fn

    jaccard=tp/(tp+fn+fp) if den_jaccard > 0 else np.nan
    precision=tp/(tp+fp) if den_precision > 0 else np.nan
    recall=tp/(tp+fn) if den_recall > 0 else np.nan
    accuracy=(tp+tn)/(tp+tn+fp+fn) if den_accuracy > 0 else np.nan
    dice= (2*tp) / (2*tp + fp + fn) if den_dice > 0 else np.nan

    out_dic = {'jaccard' : jaccard, 
               'precision' : precision,
               'recall' : recall,
               'accuracy' : accuracy, 
               'dice' : dice}
    
    return out_dic

def num2padstr(number, padding=3) : 
    padstr = str(number);
    if len(padstr) < padding :
        for ix in range(padding - len(padstr)) : 
            padstr = '0' + padstr
    
    return padstr

def compareCarpMesh(pts1, el1, pts2, el2) : 
    """
    Compare Carp Mesh 
    Returns: mean(l2_norm(pts)), median(l2_norm(elem)), comparison_code
            |'COMPARISON_POSSIBLE' : 0
    CODES = |'DIFF_NPTS'           : 1, 
            |'DIFF_NELEMS'         : 2, 
    """
    comp_codes = {'DIFF_NPTS' : 1, 
                'DIFF_NELEMS' : 2, 
                'COMPARISON_POSSIBLE' : 0}

    if (len(pts1) != len(pts2)) :
        return -1, -1, comp_codes['DIFF_NPTS']
    
    if (len(el1) != len(el2)) : 
        return -1, -1, comp_codes['DIFF_NELEMS']
     
    l2_norm_pts = l2_norm(pts1-pts2)
    l2_norm_el = l2_norm(np.array(el1)-np.array(el2))

    return np.mean(l2_norm_pts), np.mean(l2_norm_el), comp_codes['COMPARISON_POSSIBLE']


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='=', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def get_boxplot_values(data, whisker=1.5):
    low_quartile = np.nanpercentile(data, 25, method='nearest')
    high_quartile = np.nanpercentile(data, 75, method='nearest')
    iqr = high_quartile - low_quartile
    low_whis = low_quartile - whisker*iqr 
    high_whis = high_quartile + whisker*iqr 
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    midic = {
        'min': min_val,
        'low_whisker': np.max([low_whis, min_val]),
        'low_quartile': low_quartile,
        'median': np.nanmedian(data),
        'high_quartile': high_quartile,
        'high_whisker': np.min([high_whis, max_val]),
        'max': max_val
    }
    return midic

def compare_large_arrays(s0, s1, name0='s0', name1='s1') : 
    from scipy.spatial.distance import cosine 
    l2 = (s0-s1)**2 
    abs_diff = np.abs(s0-s1)
    cosine_similarity = 1 - cosine(s0, s1)
    midic = {
        'diff_square' : l2, 
        'diff_abs' : abs_diff,
        'cosine_similarity' : cosine_similarity,
        name0 : s0 ,
        name1 : s1
    }

    return midic

def classify_array(arr , thresholds: list):
    """
    Classifies an array based on given thresholds.

    Parameters:
    arr (numpy.ndarray): Array of values to classify.
    thresholds (list): List of tuples where each tuple contains the name of the classification and the corresponding threshold values.

    Returns:
    numpy.ndarray: Array of classifications.
    """
    classifications = np.zeros_like(arr, dtype=int)
    for i, value in enumerate(arr):
        for j, threshold in enumerate(thresholds):
            if threshold[0] <= value < threshold[1]:
                classifications[i] = j
                break
    return classifications

def count_values_in_ranges(arr, thresholds: list) -> dict :
    """
    Counts the number of values in each range.

    Parameters:
    arr (numpy.ndarray): Array of values to classify.
    thresholds (list): List of tuples where each tuple contains the name of the classification and the corresponding threshold values.

    Returns:
    dict: Dictionary containing the number of values in each range.
    """
    classifications = classify_array(arr, thresholds)
    counts = { }
    for ix in range(len(thresholds)):
        counts[ix] = np.count_nonzero(classifications == ix)
        
    return counts

def save_json(fname, data) : 
    import json 
    with open(fname, 'w') as f : 
        json.dump(data, f)