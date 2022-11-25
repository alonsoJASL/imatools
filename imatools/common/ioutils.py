
import glob
import os
import platform as pltf
import readline
import sys

import numpy as np


def l2_norm(a): return np.linalg.norm(a, axis=1)
def dot_prod_vec(a,b): return np.sum(a*b, axis=1)

def history(n=0):
    """
    Prints history
    """
    if n==0:
        NUM=readline.get_current_history_length()
    else:
        NUM=n

    for ix in range(NUM):
        print (readline.get_history_item(ix + 1))

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

def cout(msg, typeMsg="INFO", print2console=True):
    """
    LOGGING FUNCTION
    """
    if print2console==True:
        print("[{}] {}".format(typeMsg, msg))

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

def readParsePts(ptsFname):
    """
    Read parse CARP point files
    """
    numNodes=getTotal(ptsFname)
    nodes=np.loadtxt(ptsFname, skiprows=1)

    if (numNodes != len(nodes)):
        print("Error in file")
        exit(-2)

    return nodes, numNodes

def readParseElem(elFname):
    """
    Read and parse CARP element file
    """
    nElem, elemStr = getFileContentWithTotal(elFname)
    el = [(line.strip()).split() for line in elemStr]

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
    jaccard=tp/(tp+fn+fp) 
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    dice= (2*tp) / (2*tp + fp + fn)

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
