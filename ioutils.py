
import os, sys, subprocess, pdb, re, struct,errno
import numpy as np
import string

def fullfile(*paths):
    """
    Returns path separated by '/'
    """
    s='/'
    return s.join(paths)

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
