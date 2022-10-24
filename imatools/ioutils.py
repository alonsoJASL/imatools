
import sys
import numpy as np
import readline
import glob
import platform

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

    cout("Number of elements: {}".format(len(el)))

    elem=list()
    for e in el:
        nel = 4 if e[0]=='Tr' else 5
        elem_before = e[1:nel]
        elem.append([int(ex.strip()) for ex in elem_before])

    region_before = [e[-1] for e in el]
    region = [int(x.strip()) for x in region_before]

    return pts, elem, region


def saveToCarpTxt(pts, el, mshname):
    np.savetxt(mshname+'.pts', pts, header=str(len(pts)), comments='', fmt='%6.12f')
    np.savetxt(mshname+'.elem', el, header=str(len(el)), comments='', fmt='Tr %d %d %d 1')

def near(value1, value2, tol=1e-8) : 
    return (np.abs(value1-value2) <= tol)

