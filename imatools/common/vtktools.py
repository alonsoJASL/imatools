import sys
import os

import numpy as np
import pandas as pd
import vtk
import vtk.util.numpy_support as vtknp
import networkx as nx  # For the graph‐based connectivity method


import re
from PIL import Image
from imatools.common.config import configure_logging 

logger = configure_logging(__name__)

def parse_dotmesh_file(file_path, myencoding='utf-8'):
    """
    Parses a Biosense Webster Triangulated Mesh file and extracts relevant sections.

    Args:
        file_path (str): The path to the Triangulated Mesh file.
        myencoding (str): The encoding to use when reading the file.
                Common encodings: 'utf-8', 'latin-1', 'windows-1252', 'iso-8859-1'

    Returns:
        tuple: A tuple containing three dictionaries representing the extracted sections:
               - A dictionary containing general attributes.
               - A dictionary containing vertex data.
               - A dictionary containing triangle data.

    The function reads the contents of the specified mesh file and extracts data from
    sections denoted as 'GeneralAttributes', 'VerticesSection', and 'TrianglesSection'.
    Each section's data is stored in a separate dictionary, where keys correspond to
    identifiers or indices, and values represent the associated data.

    Example:
        file_path = 'your_mesh_file.mesh'
        general_attrs, vertices, triangles = parse_mesh_file(file_path)
        print("General Attributes:", general_attrs)
        print("Vertices Section:", vertices)
        print("Triangles Section:", triangles)
    """
    with open(file_path, 'r', encoding=myencoding) as file:
        mesh_data = file.readlines()

    # Initialize dictionaries to store parsed data
    accepatable_keys = ['MeshID', 'MeshName', 'NumVertex', 'NumTriangle']
    comments = ['#', ';', '//']
    general_attributes = {'MeshID': None, 'MeshName': 'not-set', 'NumVertex': 0, 'NumTriangle': 0}
    vertices_section = {}
    triangles_section = {}

    current_section = None

    # Define regular expressions to match section headers
    general_attr_regex = re.compile(r'\[GeneralAttributes\]')
    vertices_section_regex = re.compile(r'\[VerticesSection\]')
    triangles_section_regex = re.compile(r'\[TrianglesSection\]')

    for line in mesh_data : 
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        if line[0] in comments : 
            continue
        if general_attr_regex.match(line):
            current_section = 'GeneralAttributes'
            continue

        if '=' in line:
            key, value = line.split('=')
            if key.strip() in accepatable_keys:
                general_attributes[key.strip()] = value.strip()

    print(general_attributes)
    n_vertex = int(general_attributes['NumVertex'])
    n_triangle = int(general_attributes['NumTriangle'])

    vertices_section = {'index': np.ndarray(n_vertex, dtype=np.uint16), 
                        'points': np.ndarray((n_vertex, 3), dtype=float)}
    triangles_section = {'index': np.ndarray(n_triangle, dtype=np.uint16), 
                        'elements': np.ndarray((n_triangle, 3), dtype=np.uint16)}
    
    count_vertices = 0
    count_triangles = 0
    for line in mesh_data:
        if not line:
            continue  # Skip empty lines
        if line[0] in comments : 
            continue
        if vertices_section_regex.match(line):
            current_section = 'VerticesSection'
            continue

        if triangles_section_regex.match(line):
            current_section = 'TrianglesSection'
            continue

        if current_section == 'VerticesSection':
            if '=' in line:
                index, data = line.strip().split('=')
                vertices_section['index'][count_vertices] = np.uint16(index.strip())
                vertices_section['points'][count_vertices, :] = [float(x) for x in data.strip().split()[:3]]
                count_vertices += 1
                if count_vertices == n_vertex : 
                    continue

        if current_section == 'TrianglesSection':
            if '=' in line:
                index, data = line.strip().split('=')
                triangles_section['index'][count_triangles] = np.uint16(index.strip())
                triangles_section['elements'][count_triangles, :] = [np.uint16(x) for x in data.strip().split()[:3]]
                count_triangles += 1
                if count_triangles == n_triangle :
                    current_section = None
                    continue

    return general_attributes, vertices_section, triangles_section



def l2_norm(a): return np.linalg.norm(a, axis=1)
def dot_prod_vec(a,b): return np.sum(a*b, axis=1)

DATA_TYPES = ['polydata', 'ugrid']
def readVtk(fname, input_type='polydata'):
    logger.warning("This function is deprecated. Please use read_vtk instead.")
    return read_vtk(fname, input_type)

def read_vtk(fname, input_type='polydata'):
    """
    Read VTK file
    """
    if input_type not in DATA_TYPES:
        logger.error(f"Invalid input type: {input_type}")
        raise ValueError(f"Invalid input type: {input_type}")
    
    if input_type == 'ugrid':
        reader = vtk.vtkUnstructuredGridReader()
    else : 
        # "input_type == 'polydata'"
        reader = vtk.vtkPolyDataReader()
        
    logger.info(f"Reading VTK [{input_type}] file: {fname}")
    reader.SetFileName(fname)
    reader.Update()

    return reader.GetOutput()

def writeVtk(mesh, directory, outname="output", output_type='polydata'):
    logger.warning("This function is deprecated. Please use write_vtk instead.")
    return write_vtk(mesh, directory, outname, output_type)

def write_vtk(mesh, directory, outname="output", output_type='polydata'):
    filename = os.path.join(directory, outname)
    filename += ".vtk" if not filename.endswith(".vtk") else ""

    if output_type not in DATA_TYPES:
        logger.error(f"Invalid output type: {output_type}")
        raise ValueError(f"Invalid output type: {output_type}")
    
    if output_type == 'ugrid':
        writer = vtk.vtkUnstructuredGridWriter()
    else :
        writer=vtk.vtkPolyDataWriter()
    writer.WriteArrayMetaDataOff()
    writer.SetInputData(mesh)
    writer.SetFileName(filename)
    writer.SetFileTypeToASCII()
    # check for vtk version 
    if vtk.vtkVersion().GetVTKMajorVersion() >= 9 and vtk.vtkVersion().GetVTKMinorVersion() >= 1:
        writer.SetFileVersion(42)
    writer.Update()

def set_cell_to_point_data(msh, fieldname='scalars') : 
    c2pt = vtk.vtkCellDataToPointData()
    c2pt.SetInputData(msh)
    c2pt.PassCellDataOn()
    c2pt.SetContributingCellOption(0)
    c2pt.Update()

    omsh=c2pt.GetPolyDataOutput()
    omsh.GetPointData().GetScalars().SetName(fieldname)

    return omsh

def setCellDataToPointData(msh, fieldname='scalars') :
    """
    legacy name of function. In future, please use function: 
    set_cell_to_point_data
    """
    print(__doc__)
    return set_cell_to_point_data(msh, fieldname)

def get_cog_per_element(msh) -> np.ndarray:
    pts, el = extractPointsAndElemsFromVtk(msh)
    element_coordinates = pts[el]

    cog = np.mean(element_coordinates, axis=1)
    
    return cog

def getHausdorffDistance(input_mesh0, input_mesh1, label=0):
    """
    Get Hausdorf Distance between 2 surface meshes
    """
    hd = getHausdorffDistanceFilter(input_mesh0, input_mesh1, label)

    return hd.GetOutput()

def getHausdorffDistanceFilter(input_mesh0, input_mesh1, label=0):
    """
    Get vtkHausdorffDistancePointSetFilter output between 2 surface meshes
    """
    mesh0 = vtk.vtkPolyData()
    mesh1 = vtk.vtkPolyData()
    if label==0:
        
        mesh0.DeepCopy(input_mesh0)
        mesh1.DeepCopy(input_mesh1)
    else:
        
        mesh0=ugrid2polydata(thresholdExactValue(input_mesh0, label))
        mesh1=ugrid2polydata(thresholdExactValue(input_mesh1, label))

    hd=vtk.vtkHausdorffDistancePointSetFilter()
    hd.SetInputData(0, mesh0)
    hd.SetInputData(1, mesh1)
    hd.SetTargetDistanceMethodToPointToCell()
    hd.Update()

    return hd

def genericThreshold(msh, exactValue, typeThres='exact'):
    """
    Threshold polydata
    Returns a unstructured grid
    """
    
    thresBehaviour={'exact': 0, 'upper': 1, 'lower': 2}

    th=vtk.vtkThreshold()
    th.SetInputData(msh)

    if thresBehaviour[typeThres] == 0: # exact
        th.ThresholdBetween(exactValue,exactValue)
    elif thresBehaviour[typeThres] == 1: # upper
        th.ThresholdByUpper(exactValue)
    elif thresBehaviour[typeThres] == 2: # lower
        th.ThresholdByLower(exactValue)
    else:
        
        sys.exit(-1)

    th.Update()

    return th.GetOutput()

def thresholdExactValue(msh, exactValue):
    """
    Threshold polydata at exact value (like a tag)
    Returns a unstructured grid
    """

    return genericThreshold(msh, exactValue, 'exact')

def convertPointDataToNpArray(vtk_input, str_scalars):
    """
    Convert vtk scalar data to numpy array
    """
    vtkArrayDistance=vtk_input.GetPointData().GetScalars(str_scalars)
    distance=vtknp.vtk_to_numpy(vtkArrayDistance)

    return distance

def convertCellDataToNpArray(vtk_input, str_scalars):
    """
    Convert vtk (cell) scalar data to numpy array
    """
    vtkArrayDistance=vtk_input.GetCellData().GetScalars(str_scalars)
    distance=vtknp.vtk_to_numpy(vtkArrayDistance)

    return distance



def extractPointsAndElemsFromVtk(msh):
    pts=[list(msh.GetPoint(ix)) for ix in range(msh.GetNumberOfPoints())]
    el = [[msh.GetCell(jx).GetPointIds().GetId(ix) for ix in range(3)] for jx in range(msh.GetNumberOfCells())]

    Xpts=np.asarray(pts)
    Tri=np.asarray(el)

    return Xpts, Tri

def ugrid2polydata(ugrid):
    """
    Converts unstructured grid to polydata using the geometry filter
    """
    gf=vtk.vtkGeometryFilter()
    gf.SetInputData(ugrid)
    gf.Update()

    return gf.GetOutput()

def getSurfaceArea(msh):
    mp = vtk.vtkMassProperties();
    mp.SetInputData(msh)
    return mp.GetSurfaceArea()

def getBooleanOperationFilter(msh0, msh1, operation_str='union'):
    opts={'union': 0, 'intersection': 1, 'difference': 2}
    bopd = vtk.vtkBooleanOperationPolyDataFilter()
    bopd.SetOperation(opts[operation_str])
    bopd.SetInputData(0, msh0)
    bopd.SetInputData(1, msh1)
    bopd.Update()

    return bopd

def getBooleanOperation(msh0, msh1, operation_str='union'):
    bopd = getBooleanOperationFilter(msh0, msh1, operation_str)
    return bopd.GetOutput()

def getSurfacesJaccard(pd1, pd2):
    """
    computes the jaccard distance for two polydata objects
    """
    union = getBooleanOperation(pd1, pd2, 'union')
    intersection = getBooleanOperation(pd1, pd2, 'intersection')

    return getSurfaceArea(intersection)/getSurfaceArea(union)

def saveCarpAsVtk(pts, el, dir, name, dat=None):
    nodes = vtk.vtkPoints()
    for ix in range(len(pts)):
        nodes.InsertPoint(ix, pts[ix,0], pts[ix,1], pts[ix,2])

    elems = vtk.vtkCellArray()
    for ix in range(len(el)):
        elIdList=vtk.vtkIdList()
        elIdList.InsertNextId(el[ix][0])
        elIdList.InsertNextId(el[ix][1])
        elIdList.InsertNextId(el[ix][2])
        elems.InsertNextCell(elIdList)

    pd=vtk.vtkPolyData()
    pd.SetPoints(nodes)
    pd.SetPolys(elems)
    if dat is not None:
        pd.GetPointData().SetScalars(vtknp.numpy_to_vtk(dat))
        p2c=vtk.vtkPointDataToCellData()
        p2c.SetInputData(pd)
        p2c.Update()
        outpd=p2c.GetOutput()
    else:
        outpd=pd

    writeVtk(outpd, dir, name)

def getElemPermutation(msh0, msh1) :
    '''
    perm = getElemPermutation(msh0, msh1)
    produces perm such that
    msh0[perm] = msh1
    '''
    n0=len(msh0)
    n1=len(msh1)

    if n0!=n1 :
        
        return -1

    perm=np.zeros(n0, 1)
    # for el in msh0 :

    return 0

def projectCellData(msh_source, msh_target) : 
    """
    Projects TARGET's cell data on SOURCE
    """
    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh_source)

    target_pl = vtk.vtkCellLocator()
    target_pl.SetDataSet(msh_target)
    target_pl.AutomaticOn()
    target_pl.BuildLocator()

    target_scalar = msh_target.GetCellData().GetScalars()
    o_scalar = vtk.vtkFloatArray()
    o_scalar.SetNumberOfComponents(1)

    default_value = 0
    cog = get_cog_per_element(msh_source)
    for ix in range(msh_source.GetNumberOfCells()):
        pt = cog[ix, :]
        closest_pt = np.zeros((3, 1))
        c_id = np.int8()
        subid = np.int8()
        dist2 = np.float32()
        id_on_target = vtk.reference(c_id)
        Subid = vtk.reference(subid)
        Dist2 = vtk.reference(dist2)
        # target_pl.FindCell(pt)
        target_pl.FindClosestPoint(pt, closest_pt, id_on_target, Subid, Dist2)
        if (id_on_target > 0):
            mapped_val = target_scalar.GetTuple1(id_on_target)
        else:
            mapped_val = default_value

        o_scalar.InsertNextTuple1(mapped_val)

    omsh.GetCellData().SetScalars(o_scalar)

    return omsh

def projectPointData(msh_source, msh_target) : 
    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh_source)

    target_pl = vtk.vtkPointLocator()
    target_pl.SetDataSet(msh_target)
    target_pl.AutomaticOn()
    target_pl.BuildLocator()

    target_scalar = msh_target.GetPointData().GetScalars()
    o_scalar = vtk.vtkFloatArray()
    o_scalar.SetNumberOfComponents(1)

    default_value = 0
    for ix in range(msh_source.GetNumberOfPoints()):
        pt = msh_source.GetPoint(ix)
        id_on_target = target_pl.FindClosestPoint(pt)

        mapped_val = target_scalar.GetTuple1(id_on_target) if (
            id_on_target > 0) else default_value
        o_scalar.InsertNextTuple1(mapped_val)

    omsh.GetPointData().SetScalars(o_scalar)

    return omsh


def fibrosis_overlap(msh0, msh1, th0, th1=None, name0='msh0', name1='msh1', type='cell'):
    """Make sure msh0 aligns with msh1 in number of cells"""
    th1 = th0 if (th1 is None) else th1
    assert type in ['cell', 'point'], 'Argument "type" expected to be "cell" or "point"'

    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh0)
    o_scalar = vtk.vtkFloatArray()
    o_scalar.SetNumberOfComponents(1)

    if type=='cell' : 
        scalar0 = msh0.GetCellData().GetScalars()
        scalar1 = msh1.GetCellData().GetScalars()
    else :  
        scalar1 = msh1.GetPointData().GetScalars()
        scalar0 = msh0.GetPointData().GetScalars()

    countn = 0
    count0 = 0
    count1 = 0
    countb = 0

    total_values = msh0.GetNumberOfCells() if type == 'cell' else msh0.GetNumberOfPoints()
    for ix in range(total_values):
        value_assigned = 0

        if (scalar0.GetTuple1(ix) == 0 or scalar1.GetTuple1(ix) == 0):
            value_assigned = -1

        else:
            if (scalar0.GetTuple1(ix) >= th0):
                value_assigned += 1

            if (scalar1.GetTuple1(ix) >= th1):
                value_assigned += 2

            if (value_assigned == 0):
                countn += 1
            elif (value_assigned == 1):
                count0 += 1
            elif (value_assigned == 2):
                count1 += 1
            elif (value_assigned == 3):
                countb += 1

        o_scalar.InsertNextTuple1(value_assigned)

    if type == 'cell' : 
        omsh.GetCellData().SetScalars(o_scalar)
    else : 
        omsh.GetPointData().SetScalars(o_scalar)

    countt = countn + count0 + count1 + countb
    count_dic = {'total': countt, name0: count0,
                 name1: count1, 'overlap': countb, 'none': countn}

    return omsh, count_dic

def fibrosis_overlap_points(msh0, msh1, th0, th1=None, name0='msh0', name1='msh1') : 
    omsh, count_dic = fibrosis_overlap(msh0, msh1, th0, th1, name0, name1, type='point')
    return omsh, count_dic 

def fibrosis_overlap_cells(msh0, msh1, th0, th1=None, name0='msh0', name1='msh1') : 
    omsh, count_dic = fibrosis_overlap(msh0, msh1, th0, th1, name0, name1, type='cell')
    return omsh, count_dic 

def fibrosisOverlapCell(msh0, msh1, th0, th1=None, name0='msh0', name1='msh1') : 
    """Make sure msh0 aligns with msh1 in number of cells"""
    th1 = th0 if (th1 is None) else th1

    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh0)
    o_scalar = vtk.vtkFloatArray()
    o_scalar.SetNumberOfComponents(1)
    
    scalar0 = msh0.GetCellData().GetScalars()
    scalar1 = msh1.GetCellData().GetScalars()

    countn = 0
    count0 = 0
    count1 = 0
    countb = 0

    for ix in range(msh0.GetNumberOfCells()):
        value_assigned = 0

        if (scalar0.GetTuple1(ix) == 0 or scalar1.GetTuple1(ix) == 0):
            value_assigned = -1

        else:
            if (scalar0.GetTuple1(ix) >= th0):
                value_assigned += 1
        
            if (scalar1.GetTuple1(ix) >= th1):
                value_assigned += 2
                    
            if (value_assigned == 0) : 
                countn += 1
            elif (value_assigned == 1) : 
                count0 += 1 
            elif (value_assigned == 2) : 
                count1 += 1
            elif (value_assigned == 3) : 
                countb += 1 
            
        o_scalar.InsertNextTuple1(value_assigned)
    
    omsh.GetCellData().SetScalars(o_scalar)
    
    countt = countn + count0 + count1 + countb
    count_dic = {'total' : countt, name0 : count0, name1: count1, 'overlap' : countb, 'none' : countn}

    return omsh, count_dic

def point_to_cell_data(msh, fieldname='scalars') :
    """
    Convert point data to cell data
    """
    p2c = vtk.vtkPointDataToCellData()
    p2c.SetInputData(msh)
    p2c.PassPointDataOn()
    p2c.Update()

    omsh = p2c.GetOutput()
    omsh.GetCellData().GetScalars().SetName(fieldname)

    return omsh

def cell_to_point_data(msh, fieldname='scalars') :
    """
    Convert cell data to point data
    """
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(msh)
    c2p.PassCellDataOn()
    c2p.Update()

    omsh = c2p.GetOutput()
    omsh.GetPointData().GetScalars().SetName(fieldname)

    return omsh

def fibrosis_score(msh, th, type='cell') : 
    """Assumes the scalars in msh have been normalised"""

    assert type in ['cell', 'point'], 'Argument "type" expected to be "cell" or "point"'
    if type == 'cell' : 
        scalars = msh.GetCellData().GetScalars()
    else :
        scalars = msh.GetPointData().GetScalars()

    total_values = msh.GetNumberOfCells() if type=='cell' else msh.GetNumberOfPoints()
    countt = float(total_values)
    countfib = 0.0

    for ix in range(total_values):
        if (scalars.GetTuple1(ix) == 0):
            countt -= 1.0

        elif (scalars.GetTuple1(ix) >= th):
            countfib += 1.0

    return countfib/countt

def fibrosis_score_cell(msh, th):
    return fibrosis_score(msh, th, 'cell')


def fibrosis_score_point(msh, th):
    return fibrosis_score(msh, th, 'point')

def fibrorisScore(msh, th) : 
    """Assumes the scalars in msh have been normalised"""
    scalars = msh.GetCellData().GetScalars()
    countt = float(msh.GetNumberOfCells())
    countfib = 0.0

    for ix in range(msh.GetNumberOfCells()):
        if (scalars.GetTuple1(ix) == 0) : 
            countt -= 1.0
        
        elif (scalars.GetTuple1(ix) >= th) : 
            countfib += 1.0
    
    return countfib/countt

norm2 = lambda a : np.linalg.norm(a)
norm_vec = lambda a : a/norm2(a)

def compare_fibres(msh_a, msh_b, f_a , f_b) :  
    tot_left = msh_a.GetNumberOfPoints()
    tot_right = msh_b.GetNumberOfPoints()

    msh0=vtk.vtkPolyData()
    msh1=vtk.vtkPolyData()
    if (tot_left >= tot_right) : 
        msh0.DeepCopy(msh_a) 
        msh1.DeepCopy(msh_b) 
        f0 = f_a
        f1 = f_b
    else :
        msh0.DeepCopy(msh_b) 
        msh1.DeepCopy(msh_a) 
        f0 = f_b
        f1 = f_a
    
    # pts1, el1 = extractPointsAndElemsFromVtk(msh1)
    cog1 = get_cog_per_element(msh1)

    cell_loc=vtk.vtkCellLocator()
    cell_loc.SetDataSet(msh0)
    cell_loc.BuildLocator()

    num_elements1 = len(cog1)
    f0v1_dot = np.zeros(num_elements1)
    f0v1_dist = np.zeros(num_elements1)
    centres = np.zeros(num_elements1)

    norm_vec_f0 = np.divide(f0.T,np.linalg.norm(f0, axis=1)).T 
    norm_vec_f1 = np.divide(f1.T,np.linalg.norm(f1, axis=1)).T 

    region1 = convertCellDataToNpArray(msh1, 'elemTag')

    for jx in range(num_elements1) : 
        cellId = vtk.reference(0)
        c = [0.0, 0.0, 0.0]
        subId = vtk.reference(0)
        d = vtk.reference(0.0)

        cell_loc.FindClosestPoint(cog1[jx], c, cellId, subId, d)
        centres[jx] = c
        a=norm_vec_f0[cellId.get()]
        b=norm_vec_f1[jx]
        f0v1_dot[jx] = np.dot(a,b)

    f0v1_dist = np.linalg.norm(cog1 - c, axis=1)
    f0v1_angles = np.arccos(f0v1_dot)
    f0v1_abs_dot = np.abs(f0v1_dot)
    f0v1_angle_abs_dot = np.arccos(f0v1_abs_dot)

    d={'region' : region1,
        'dot_product' : f0v1_dot,
        'angle' : f0v1_angles,
        'distance_to_point' : f0v1_dist,
        'abs_dot_product' : f0v1_abs_dot,
        'angle_from_absdot' : f0v1_angle_abs_dot}
    
    return pd.DataFrame(data=d)

def compare_mesh_sizes(msh_left_name, msh_right_name, left_id, right_id, map_type_id) : 
    msh_left = readVtk(msh_left_name)
    msh_right = readVtk(msh_right_name)

    if (map_type_id == 1) : # elem  
        tot_left = msh_left.GetNumberOfCells()
        tot_right = msh_right.GetNumberOfCells() 
    elif (map_type_id == 0 ) : # pts
        tot_left = msh_left.GetNumberOfPoints()
        tot_right = msh_right.GetNumberOfPoints() 
    else : 
        return None, None, None, None, None, None

    path_large = msh_left_name  # 0
    path_small = msh_right_name # 1
    tot_large = tot_left
    tot_small = tot_right
    large_id = left_id
    small_id = right_id

    if tot_left < tot_right : 
        path_large = msh_right_name
        path_small = msh_left_name
        tot_large = tot_right
        tot_small = tot_left
        large_id = right_id
        small_id = left_id
    
    return path_large, path_small, tot_large, tot_small, large_id, small_id

def map_cells(msh_large, cog_small, tot_small, large_id, small_id) : 
    cell_locate_on_large=vtk.vtkCellLocator()
    cell_locate_on_large.SetDataSet(msh_large)
    cell_locate_on_large.BuildLocator()

    cell_ids_small = np.arange(tot_small)
    cell_ids_large = np.zeros(tot_small, dtype=int)
    l2_norm_filter = np.zeros(tot_small, dtype=float)
    
    pts_in_large = np.zeros((tot_small, 3), dtype=float)

    for ix in range(tot_small): # tot_small
        cellId_in_large = vtk.reference(0)
        c_in_large = [0.0, 0.0, 0.0]
        dist_to_large = vtk.reference(0.0)

        cell_locate_on_large.FindClosestPoint(cog_small[ix], c_in_large, cellId_in_large, vtk.reference(0), dist_to_large)
        cell_ids_large[ix] = cellId_in_large.get()

        l2_norm_filter[ix] = dist_to_large

        pts_in_large[ix, 0] = c_in_large[0]
        pts_in_large[ix, 1] = c_in_large[1]
        pts_in_large[ix, 2] = c_in_large[2]

    mapping_dictionary_cells = {
        small_id : cell_ids_small, 
        large_id : cell_ids_large, 
        'distance_manual' : l2_norm(cog_small - pts_in_large), 
        'distance_auto'  : l2_norm_filter, 
        'X_'+small_id.lower() : cog_small[:, 0], 
        'Y_'+small_id.lower() : cog_small[:, 1], 
        'Z_'+small_id.lower() : cog_small[:, 2],
        'X_'+large_id.lower() : pts_in_large[:, 0],
        'Y_'+large_id.lower() : pts_in_large[:, 1],
        'Z_'+large_id.lower() : pts_in_large[:, 2]
    }

    return mapping_dictionary_cells

def map_points(msh_large, msh_small, large_id, small_id) : 

    tot_small = msh_small.GetNumberOfPoints()

    pts_locate_on_large = vtk.vtkPointLocator()
    pts_locate_on_large.SetDataSet(msh_large)
    pts_locate_on_large.BuildLocator()

    pts_ids_small = np.arange(tot_small)
    pts_ids_large = np.zeros(tot_small, dtype=int)

    pts_small = np.zeros((tot_small,3), dtype=float)
    pts_large = np.zeros((tot_small, 3), dtype=float)

    for ix in range(tot_small) : 
        pt_small = np.asarray(msh_small.GetPoint(ix))
        
        ptsId_in_large = pts_locate_on_large.FindClosestPoint(pt_small)
        pts_ids_large[ix] = ptsId_in_large

        pt_large = np.asarray(msh_large.GetPoint(ptsId_in_large))

        pts_small[ix, 0] = pt_small[0]
        pts_small[ix, 1] = pt_small[1]
        pts_small[ix, 2] = pt_small[2]

        pts_large[ix, 0] = pt_large[0]
        pts_large[ix, 1] = pt_large[1]
        pts_large[ix, 2] = pt_large[2]

    mapping_dictionary_points = {
        small_id : pts_ids_small, 
        large_id : pts_ids_large, 
        'distance_manual' : l2_norm(pts_small-pts_large), 
        'X_'+small_id.lower() : pts_small[:, 0], 
        'Y_'+small_id.lower() : pts_small[:, 1], 
        'Z_'+small_id.lower() : pts_small[:, 2], 
        'X_'+large_id.lower() : pts_large[:, 0],
        'Y_'+large_id.lower() : pts_large[:, 1],
        'Z_'+large_id.lower() : pts_large[:, 2]
    }

    return mapping_dictionary_points

def create_mapping(msh_left_name, msh_right_name, left_id, right_id, map_type='elem') :
    """ 
    Create mapping to closest [PTS|ELEMS] from msh_left to msh_right 
    """
    map_dic = {'pts' : 0, 'elem' : 1}
    map_id = map_dic[map_type]
    path_large, path_small, _, tot_small, large_id, small_id = compare_mesh_sizes(msh_left_name, msh_right_name, left_id, right_id, map_id)

    if (path_large is None) : 
        print('ERROR: Wrong mapping type { elem, pts }')
        return None

    msh_large = readVtk(path_large) # 0
    msh_small = readVtk(path_small) # 1

    cog_small = global_centre_of_mass(msh_small)
    cog_large = global_centre_of_mass(msh_large) 
    
    if (norm2(cog_small - cog_large) > 1) :
        logger.warning('WARNING: Meshes are not aligned. Translating to (0,0,0)')
        msh_large = translate_to_point(msh_large)
        msh_small = translate_to_point(msh_small)

    if (map_id == 1) : # elem 
        elem_cog_small = get_cog_per_element(msh_small)
        mapping_dictionary = map_cells(msh_large, elem_cog_small, tot_small, large_id, small_id) 
    else : 
        elem_cog_small = [msh_small.GetPoint(ix) for ix in range(tot_small)]
        elem_cog_small = np.asarray(elem_cog_small) 
        mapping_dictionary = map_points(msh_large, msh_small, large_id, small_id)

    return mapping_dictionary

def convert_5_to_4(imsh, omsh) :
    """
    Input: msh paths
        imsh (input)
        omsh (output)
    """
    # Read the VTK legacy file in version 5 format
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(imsh)
    reader.Update()
    print(reader.GetFileVersion())

    # Convert the file to VTK legacy format
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(reader.GetOutput())
    writer.SetFileName(omsh)
    writer.SetFileTypeToASCII()
    # writer.SetHeader("vtk version 4.0")
    writer.SetFileVersion(42)
    writer.Update()

def flip_xy(polydata) :
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()

    for i in range(num_points):
        original_coords = points.GetPoint(i)
        modified_coords = [-original_coords[0], -original_coords[1], original_coords[2]]
        points.SetPoint(i, modified_coords)

    polydata.Modified()

def global_centre_of_mass(mesh) :
    """Return the centre of mass of a mesh"""
    pts, _ = extractPointsAndElemsFromVtk(mesh)
    return np.mean(pts, axis=0)

def translate_to_point(mesh, point=[0,0,0]) :
    """Translate a mesh to a point"""
    cog = global_centre_of_mass(mesh)
    transform = vtk.vtkTransform()
    transform.Translate(point[0]-cog[0], point[1]-cog[1], point[2]-cog[2]) 

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(mesh)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    return transform_filter.GetOutput()

def get_filtered_array(df: pd.DataFrame, field: str, mesh_path: str, mesh_scalar_field='scalars', dist_field='distance_manual', max_distance=1.0) -> np.ndarray:
    # filter the DataFrame by distance_manual
    df_filtered = df[df[dist_field] < max_distance] 
    mesh_indices = df_filtered[field].values

    # read the mesh and extract the scalar data
    msh = readVtk(mesh_path)
    mesh_array = convertCellDataToNpArray(msh, mesh_scalar_field)
    mesh_array = mesh_array[mesh_indices]
    
    return mesh_array, df_filtered

def np_to_vtk_array(data: np.ndarray, name: str) -> vtk.vtkFloatArray:
    vtk_array = vtknp.numpy_to_vtk(data)
    vtk_array.SetName(name)

    return vtk_array

def mask_cell_scalars(msh, values, indices, scalar_field='scalars') : 
    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh)
    array = convertCellDataToNpArray(omsh, scalar_field)

    if len(indices) != len(values) : 
        raise ValueError('Indices and values must have the same length')
    
    for ix, indx in enumerate(indices) :
        array[indx] = values[ix]

    omsh.GetCellData().SetScalars(np_to_vtk_array(array, scalar_field))
    return omsh

    
def set_vtk_scalars(msh, array, indices = None) -> vtk.vtkPolyData: 
    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh)
    
    # Check if 'scalars' array exists
    scalars = omsh.GetCellData().GetScalars()
    if scalars is None:
        # If 'scalars' array does not exist, create it
        scalars = vtk.vtkFloatArray()
        scalars.SetName('scalars')
        omsh.GetCellData().SetScalars(scalars)

    inav_array = -1*np.ones_like(convertCellDataToNpArray(omsh, 'scalars')) 

    if indices is not None :
        inav_array[indices] = array
    else: 
        inav_array = array

    omsh.GetCellData().SetScalars(np_to_vtk_array(inav_array, 'scalars'))
    return omsh

def indices_at_scalar(msh, scalar=0.0, fieldname='scalars') -> np.ndarray:
    scalars = convertCellDataToNpArray(msh, fieldname)
    indices = np.where(scalars == scalar)[0]

    return indices

def verify_cell_indices(msh, test_indices, test_locations) : 
    """
    Verifies that the test_indices in the mesh (msh) 
    are the same as the test_locations.

    test_locations is linked to test_indices via the
    centre of gravity of the mesh elements.
    """

    cog = get_cog_per_element(msh)
    test_cog = cog[test_indices, :]
    diff = np.linalg.norm(test_cog - test_locations, axis=1)

    return np.sum(diff)


def verify_cell_indices_from_mesh(msh1, msh_test, test_indices) :
    """
    Verifies that the test_indices in the mesh (msh) 
    are the same as the test_locations.

    test_locations is linked to test_indices via the
    centre of gravity of the mesh elements.
    """

    cog_test = get_cog_per_element(msh_test)
    test_cog = cog_test[test_indices, :]
   
    return verify_cell_indices(msh1, test_indices, test_cog)

def flip_xy(polydata) :
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()

    for i in range(num_points):
        original_coords = points.GetPoint(i)
        modified_coords = [-original_coords[0], -original_coords[1], original_coords[2]]
        points.SetPoint(i, modified_coords)

    polydata.Modified()

def join_vtk(msh0, msh1) :
    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(msh0)
    appendFilter.AddInputData(msh1)
    appendFilter.Update()

    return appendFilter.GetOutput()

def set_cell_scalars(vtklabel, label) : 
    """
    Set the cell scalars of a vtkPolyData object
    """
    return set_vtk_scalars(vtklabel, np.ones(vtklabel.GetNumberOfCells())*label)

def vtk_from_points_file(file_path:str, mydelim=',') : 
    """
    Creates a vtkPolyData object from a points file
    """
    points_read = np.loadtxt(file_path, delimiter=mydelim)
    points = vtk.vtkPoints()

    for pt in points_read : 
        points.InsertNextPoint(pt[0], pt[1], pt[2]) 
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    return polydata

def export_as(input_mesh, output_file: str, export_as='ply') -> None:
    """
    Export a vtkPolyData object to a file
    """
    if export_as == 'ply':
        writer = vtk.vtkPLYWriter()
    elif export_as == 'stl':
        writer = vtk.vtkSTLWriter()
    elif export_as == 'obj':
        writer = vtk.vtkOBJWriter()
    elif export_as == 'vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        raise ValueError('Export format not supported')

    writer.SetFileName(output_file)
    writer.SetInputData(input_mesh)
    writer.Write()


    import vtk

def render_vtk_to_png(vtk_files, output_dir, window_size=(800, 600)):
    """
    Renders VTK mesh files to PNG images and saves them in the specified output directory.

    Parameters:
        vtk_files (list of str): List of paths to VTK files.
        output_dir (str): Directory where PNG images will be saved.

    Returns:
        None
    """
    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(window_size)  # Set window size as needed

    for vtk_file in vtk_files:
        # Read the VTK file
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtk_file)
        reader.Update()
        polydata = reader.GetOutput()

        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Add the actor to the renderer
        renderer.AddActor(actor)
        renderer.SetBackground(1, 1, 1)  # Set background color to white

        # Render and create an image filter
        render_window.Render()
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.Update()

        # Write the image to a PNG file
        writer = vtk.vtkPNGWriter()
        output_filename = f"{output_dir}/{vtk_file.split('/')[-1].replace('.vtk', '.png')}"
        writer.SetFileName(output_filename)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

        # Clear the renderer for the next iteration
        renderer.RemoveAllViewProps()

def render_vtk_to_single_png(vtk_files, output_filename, grid_size=(1, 1), window_size=(800, 600), scalar_name='elemTag', input_type='ugrid', overlapping_margin=0.0, names=None):
    """
    Renders VTK mesh files to a single PNG image arranged in a grid layout.

    Parameters:
        vtk_files (list of str): List of paths to VTK files.
        output_filename (str): Path where the combined PNG image will be saved.
        grid_size (tuple of int): Number of rows and columns in the grid (rows, columns).
        window_size (tuple of int): Size of each individual render window (width, height).

    Returns:
        None
    """
    num_files = len(vtk_files)
    rows, cols = grid_size

    # Ensure the grid is large enough to fit all files
    print(f'Grid size: {rows}x{cols} / Number of files: {num_files}')
    if rows * cols < num_files:
        raise ValueError("Grid size is too small to fit all VTK files.")

    # Create a renderer, render window, and interactor
    renderers = []
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(window_size[0] * cols, window_size[1] * rows)

    for i, vtk_file in enumerate(vtk_files):
        # Read the VTK file
        if input_type == 'ugrid':
            reader = vtk.vtkUnstructuredGridReader()
        else : 
            # "input_type == 'polydata'"
            reader = vtk.vtkPolyDataReader()
        
        reader.SetFileName(vtk_file)
        reader.Update()
        ugrid = reader.GetOutput()

        # Check if scalar_name exists in the cell data
        cell_data = ugrid.GetCellData()
        point_data = ugrid.GetPointData()
        scalar_range = None
        set_scalar_range = True
        using_point_data = False
        if cell_data.HasArray(scalar_name):
            elem_tag = cell_data.GetArray(scalar_name)
            scalar_range = elem_tag.GetRange()
        elif point_data.HasArray(scalar_name):
            elem_tag = point_data.GetArray(scalar_name)
            scalar_range = elem_tag.GetRange()
            using_point_data = True
        else:
            # available_fields = [cell_data.GetArrayName(i) for i in range(cell_data.GetNumberOfArrays())]
            # print(f"File {vtk_file} does not have a '{scalar_name}' scalar field. Available fields: {available_fields}")
            # print('Continuing without coloring the mesh.')
            set_scalar_range = False

        # Create a mapper
        if input_type == 'ugrid':
            mapper = vtk.vtkDataSetMapper()
        else :
            mapper = vtk.vtkPolyDataMapper()

        mapper.SetInputData(ugrid)

        if set_scalar_range :
            # Create a lookup table to map scalar values to colors
            lut = vtk.vtkLookupTable()
            lut.SetTableRange(scalar_range)
            lut.Build()

            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(scalar_range)
            if using_point_data:
                mapper.SetScalarModeToUsePointData()
            else:
                mapper.SetScalarModeToUseCellData()
            mapper.SelectColorArray(scalar_name)
            
        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create a renderer for this actor
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(1, 1, 1)  # Set background color to white

        # Add text to the renderer
        if names and i < len(names):
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(names[i])
            text_actor.GetTextProperty().SetFontSize(24)
            text_actor.GetTextProperty().SetColor(0, 0, 0)  # Set text color to black
            text_actor.SetPosition(10, 10)  # Set text position
            renderer.AddActor2D(text_actor)

        # Calculate viewport for this renderer
        margin = -overlapping_margin  # Adjust this value to control the overlap
        row = i // cols
        col = i % cols
        viewport = [
            col / cols - margin,               # xmin
            1 - (row + 1) / rows + margin,     # ymin
            (col + 1) / cols + margin,         # xmax
            1 - row / rows - margin            # ymax
        ]
        print(f'Processing file {i+1}/{num_files}... [{row},{col}] - File: {vtk_file} ')
        renderer.SetViewport(viewport)

        # Add the renderer to the render window
        render_window.AddRenderer(renderer)
        renderers.append(renderer)

    # Render and create an image filter
    render_window.Render()
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()
    width, height, _ = vtk_image.GetDimensions()

    vtk_array = vtk_image.GetPointData().GetScalars()
    image_array = vtk.util.numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, -1)
    image_array = np.flipud(image_array)  # Flip the image vertically

    # Create a PIL image and save it
    image = Image.fromarray(image_array)
    image.save(output_filename)

def normalise_vtk_values(imsh, fieldname='scalars') :
    """
    Normalise the values of a vtkPolyData object
    """
    array = convertPointDataToNpArray(imsh, fieldname)
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    scalars = np_to_vtk_array(array, fieldname)
    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(imsh)
    omsh.GetPointData().SetScalars(scalars)
    # .GetPointData().SetScalars(scalars)
    return omsh

def convertToCarto(vtkpoly_path:str, cell_scalar_field:str, output_file:str) -> None:
    """
    Convert a vtkPolyData object to a Carto object
    """    
    try: 
        vtkpoly = readVtk(vtkpoly_path)
        working_msh = set_cell_to_point_data(vtkpoly, cell_scalar_field)
        norm_working_msh = normalise_vtk_values(working_msh, cell_scalar_field) 
        # save 
        odir = os.path.dirname(output_file)
       
        writeVtk(norm_working_msh, odir, f'normalised_{cell_scalar_field}.vtk')

        ## change lookup table for norm_working_msh
        

        lut = vtk.vtkColorTransferFunction()
        lut.SetColorSpaceToRGB()
        lut.AddRGBPoint(0.0, 0.04, 0.21, 0.25)
        lut.AddRGBPoint(0.5, 0.94, 0.47, 0.12)
        lut.AddRGBPoint(1.0, 0.90, 0.11, 0.14)
        lut.SetScaleToLinear()


    except Exception as e:
        print(f'Error: {e}')
        return
    
    with open(output_file, 'w') as cartoFile:
        # Header
        cartoFile.write("# vtk DataFile Version 3.0\n")
        cartoFile.write("PatientData Anon Anon 00000000\n")
        cartoFile.write("ASCII\n")
        cartoFile.write("DATASET POLYDATA\n")

        # Points
        cartoFile.write(f"POINTS\t{working_msh.GetNumberOfPoints()} float\n")
        points = working_msh.GetPoints()
        for ix in range(working_msh.GetNumberOfPoints()):
            pt = points.GetPoint(ix)
            cartoFile.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
        
        cartoFile.write("\n")

        # Cells 
        cartoFile.write(f"POLYGONS\t{working_msh.GetNumberOfCells()}\t{working_msh.GetNumberOfCells()*4}\n")
        for ix in range(working_msh.GetNumberOfCells()):
            cell = working_msh.GetCell(ix)
            cell_type = cell.GetCellType()
            num_points = cell.GetNumberOfPoints()
            cartoFile.write(f"{num_points}\n")
            for jx in range(num_points):
                cartoFile.write(f"{cell.GetPointId(jx)}\n")
        
        cartoFile.write("\n")

        # Scalars
        cartoFile.write(f"POINT_DATA\tSCALARS {cell_scalar_field} float\n")
        cartoFile.write("LOOKUP_TABLE lookup_table\n")
        
        scalars = working_msh.GetPointData().GetScalars()
        max_scalar = np.max(scalars)
        min_scalar = np.min(scalars)

        for kx in range(working_msh.GetNumberOfPoints()):
            value = scalars.GetTuple1(kx)
            normalized_value = (value - min_scalar) / (max_scalar - min_scalar)
            # set precision to 2 decimal places
            cartoFile.write(f"{normalized_value:.2f}\n")

        cartoFile.write("\n")

        # LUT
        numCols = 256
        cartoFile.write(f"LOOKUP_TABLE lookup_table {numCols}\n")
        lut = vtk.vtkColorTransferFunction()
        lut.SetColorSpaceToRGB()
        lut.AddRGBPoint(0.0, 0.04, 0.21, 0.25)
        lut.AddRGBPoint((numCols - 1.0) / 2.0, 0.94, 0.47, 0.12)
        lut.AddRGBPoint((numCols - 1.0), 0.90, 0.11, 0.14)
        lut.SetScaleToLinear()
        for i in range(numCols):
            color = lut.GetColor(i)
            cartoFile.write(f"{color[0]} {color[1]} {color[2]} 1.0\n")

def poly2nx(msh: vtk.vtkPolyData) -> nx.Graph: 
    G = nx.Graph()

    # Get the points (nodes)
    points = msh.GetPoints()
    if points is None:
        raise ValueError("No points found in vtkPolyData")

    num_points = points.GetNumberOfPoints()
    
    # Add nodes with positions
    for i in range(num_points):
        coord = points.GetPoint(i)  # Get (x, y, z) coordinates
        G.add_node(i, pos=np.array(coord))  # Store coordinates as node attribute

    # Get the cells (edges)
    for i in range(msh.GetNumberOfCells()):
        cell = msh.GetCell(i)
        point_ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        
        # Add edges based on cell connectivity
        for j in range(len(point_ids) - 1):  # Connect sequential points
            G.add_edge(point_ids[j], point_ids[j + 1])
        
        # If the cell is a polygon, close the loop
        if cell.GetNumberOfPoints() > 2:
            G.add_edge(point_ids[-1], point_ids[0])

    return G

def compute_cell_neighbor_count(polydata: vtk.vtkPolyData) -> vtk.vtkIntArray:
    """
    Build a graph of cell connectivity for the input polydata (assumed to consist of triangles)
    using vertex-based connectivity. Then create a vtkIntArray scalar field where each cell’s value
    is the number of its unique immediate neighbors (cells that share at least one vertex).

    Parameters:
      polydata: vtk.vtkPolyData representing a surface mesh.

    Returns:
      neighborCountArray: vtkIntArray with one component per cell, containing the number
                          of immediate neighbors.
    """
    numCells = polydata.GetNumberOfCells()
    
    # Build a dictionary mapping vertex IDs to the set of cell IDs that include that vertex.
    vertex2cells = {}
    for cellId in range(numCells):
        cell = polydata.GetCell(cellId)
        ptIds = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        for pt in ptIds:
            if pt not in vertex2cells:
                vertex2cells[pt] = set()
            vertex2cells[pt].add(cellId)
    
    # Create an undirected graph where each node represents a cell.
    G = nx.Graph()
    G.add_nodes_from(range(numCells))
    
    # For every vertex, connect all cells that share that vertex.
    for pt, cells in vertex2cells.items():
        cells_list = list(cells)
        for i in range(len(cells_list)):
            for j in range(i + 1, len(cells_list)):
                G.add_edge(cells_list[i], cells_list[j])
    
    # Create a vtkIntArray to store the number of immediate neighbors for each cell.
    neighborCountArray = vtk.vtkIntArray()
    neighborCountArray.SetName("CellNeighborCount")
    neighborCountArray.SetNumberOfComponents(1)
    neighborCountArray.SetNumberOfTuples(numCells)
    
    # For each cell, the degree in the graph is the number of immediate neighbors.
    for cellId in range(numCells):
        count = G.degree(cellId)
        neighborCountArray.SetTuple1(cellId, count)
    
    return neighborCountArray


def detect_bridges_with_graph_vertex(polydata: vtk.vtkPolyData) -> vtk.vtkIntArray:
    """
    Analyze a triangle mesh (polydata) by building a connectivity graph of its cells
    and detecting bridges. A bridge is an edge whose removal disconnects the graph.
    This version includes both edge-based and vertex-based connectivity.

    Parameters:
      polydata: vtkPolyData representing a surface mesh (assumed to be composed of triangles).

    Returns:
      bridgeArray: vtkIntArray with 1 for cells that are flagged as being part of a bridge,
                   and 0 otherwise.
    """
    numCells = polydata.GetNumberOfCells()
    edge2cells = {}  # Maps edges (sorted tuples of point IDs) to the list of cell IDs that share them.
    vertex2cells = {}  # Maps vertices (point IDs) to the list of cell IDs that contain them.

    # Step 1: Populate edge and vertex connectivity
    for cellId in range(numCells):
        cell = polydata.GetCell(cellId)
        ptIds = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        if len(ptIds) != 3:
            continue

        # Store edges (ensuring they are sorted to prevent duplicates)
        edges = [
            tuple(sorted((ptIds[0], ptIds[1]))),
            tuple(sorted((ptIds[1], ptIds[2]))),
            tuple(sorted((ptIds[2], ptIds[0])))
        ]
        for edge in edges:
            if edge not in edge2cells:
                edge2cells[edge] = []
            edge2cells[edge].append(cellId)

        # Store vertex-based connectivity
        for pt in ptIds:
            if pt not in vertex2cells:
                vertex2cells[pt] = []
            vertex2cells[pt].append(cellId)

    # Step 2: Build the connectivity graph (G)
    G = nx.Graph()
    G.add_nodes_from(range(numCells))

    # Edge-based connectivity: Connect triangles that share an edge
    for edge, cells in edge2cells.items():
        if len(cells) == 2:  # Only consider edges shared by exactly 2 triangles
            G.add_edge(cells[0], cells[1])

    # Vertex-based connectivity: Connect triangles that share at least one vertex
    for pt, cells in vertex2cells.items():
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):  # Connect all triangles that share this vertex
                G.add_edge(cells[i], cells[j])

    # Step 3: Detect bridges (edges whose removal would disconnect the graph)
    bridges = list(nx.bridges(G))

    # Step 4: Create an array to flag cells that are part of any bridge
    bridgeFlag = vtk.vtkIntArray()
    bridgeFlag.SetName("BridgeFlag")
    bridgeFlag.SetNumberOfComponents(1)
    bridgeFlag.SetNumberOfTuples(numCells)

    # Initialize all cells to 0 (not part of a bridge)
    for i in range(numCells):
        bridgeFlag.SetTuple1(i, 0)

    # Mark cells that are part of a bridge
    for (cellA, cellB) in bridges:
        bridgeFlag.SetTuple1(cellA, 1)
        bridgeFlag.SetTuple1(cellB, 1)

    return bridgeFlag


def detect_bridges_with_graph(polydata: vtk.vtkPolyData) -> vtk.vtkIntArray:
    """
    Analyze a triangle mesh (polydata) by building a connectivity graph of its cells
    and detecting bridges. A bridge is an edge whose removal disconnects the graph.
    Returns a vtkIntArray (with one entry per cell) that flags cells involved in at least one bridge.
    """
    numCells = polydata.GetNumberOfCells()
    # Map each edge (as a sorted tuple of point IDs) to the list of cell IDs that share it.
    edge2cells = {}
    for cellId in range(numCells):
        cell = polydata.GetCell(cellId)
        # Assuming triangles. Get the point IDs.
        ptIds = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        if len(ptIds) != 3:
            continue
        # For each edge (3 per triangle)
        edges = [
            tuple(sorted((ptIds[0], ptIds[1]))),
            tuple(sorted((ptIds[1], ptIds[2]))),
            tuple(sorted((ptIds[2], ptIds[0])))
        ]
        for edge in edges:
            if edge not in edge2cells:
                edge2cells[edge] = []
            edge2cells[edge].append(cellId)
    
    # Build a graph where each node is a cell (triangle) and an edge exists if two cells share an edge.
    G = nx.Graph()
    G.add_nodes_from(range(numCells))
    for edge, cells in edge2cells.items():
        if len(cells) == 2:
            G.add_edge(cells[0], cells[1])
    
    # Identify bridge edges using networkx.
    bridges = list(nx.bridges(G))
    
    # Create an array to flag cells that are part of any bridge edge.
    bridgeFlag = vtk.vtkIntArray()
    bridgeFlag.SetName("BridgeFlag")
    bridgeFlag.SetNumberOfComponents(1)
    bridgeFlag.SetNumberOfTuples(numCells)
    for i in range(numCells):
        bridgeFlag.SetTuple1(i, 0)
    
    # Mark both cells for each bridge edge.
    for (cellA, cellB) in bridges:
        bridgeFlag.SetTuple1(cellA, 1)
        bridgeFlag.SetTuple1(cellB, 1)
    
    return bridgeFlag


def detect_bridges_with_thickness(polydata: vtk.vtkPolyData, max_distance=5.0, thickness_threshold=1.5, output_raw_thickness=True) -> vtk.vtkDoubleArray:
    """
    Compute a local thickness at each vertex by casting a ray in the direction opposite the vertex normal.
    Then, flag cells that have at least one vertex with a local thickness below thickness_threshold.
    
    Parameters:
      polydata: vtkPolyData representing the surface. It is assumed that normals are computed.
      max_distance: Maximum distance to search along the ray.
      thickness_threshold: If the computed local thickness is below this threshold, the vertex is flagged.
    
    Returns:
      thicknessFlag: vtkIntArray (one entry per cell) with 1 for cells that are likely part of a narrow bridge.
    """
    # Ensure that normals exist. If not, compute them.
    if not polydata.GetPointData().GetNormals():
        normalsFilter = vtk.vtkPolyDataNormals()
        normalsFilter.SetInputData(polydata)
        normalsFilter.ComputePointNormalsOn()
        normalsFilter.ComputeCellNormalsOff()
        normalsFilter.Update()
        polydata = normalsFilter.GetOutput()

    numPoints = polydata.GetNumberOfPoints()
    
    # Build a locator for fast intersection queries.
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(polydata)
    cellLocator.BuildLocator()
    
    # Create an array to store local thickness for each vertex.
    thicknessArray = vtk.vtkDoubleArray()
    thicknessArray.SetName("LocalThickness")
    thicknessArray.SetNumberOfComponents(1)
    thicknessArray.SetNumberOfTuples(numPoints)
    
    # For each vertex, cast a ray opposite to the normal and measure distance to the next intersection.
    # We use a small epsilon to avoid detecting the originating cell.
    epsilon = 1e-6
    for i in range(numPoints):
        point = polydata.GetPoint(i)
        normal = polydata.GetPointData().GetNormals().GetTuple(i)
        # Create a ray: from the point to (point - normal * max_distance)
        start = point
        end = (point[0] - normal[0]*max_distance,
               point[1] - normal[1]*max_distance,
               point[2] - normal[2]*max_distance)
        t = vtk.mutable(0.0)
        x = [0.0, 0.0, 0.0]
        pcoords = [0.0, 0.0, 0.0]
        subId = vtk.mutable(0)
        # IntersectWithLine returns 1 if an intersection is found.
        if cellLocator.IntersectWithLine(start, end, epsilon, t, x, pcoords, subId):
            # t is a normalized parameter along the line. Multiply by max_distance.
            distance = t * max_distance
        else:
            # No intersection found: set thickness to max_distance.
            distance = max_distance
        thicknessArray.SetTuple1(i, distance)
    
    # Now, flag cells that have any vertex with thickness below the threshold.
    numCells = polydata.GetNumberOfCells()
    thicknessFlag = vtk.vtkDoubleArray()
    thicknessFlag.SetName("BridgeByThickness")
    thicknessFlag.SetNumberOfComponents(1)
    thicknessFlag.SetNumberOfTuples(numCells)
    for cellId in range(numCells):
        cell = polydata.GetCell(cellId)
        flag = 0
        raw_flag = 0
        for j in range(cell.GetNumberOfPoints()):
            ptId = cell.GetPointId(j)
            raw_flag += thicknessArray.GetTuple1(ptId)
            if thicknessArray.GetTuple1(ptId) > thickness_threshold:
                flag = 1
                break
        raw_flag /= cell.GetNumberOfPoints()
        
        if output_raw_thickness:
            thicknessFlag.SetTuple1(cellId, raw_flag)
        else:
            thicknessFlag.SetTuple1(cellId, flag)

    
    return thicknessFlag


def extract_single_label(msh: vtk.vtkPolyData, label:int, scalar_field='elemTag') -> vtk.vtkPolyData : 
    """
    Extracts a single label from a vtkPolyData object.

    Parameters:
        msh (vtk.vtkPolyData): The input surface mesh.
        label (int): The label to extract.
        scalar_field (str): The scalar field name to use for extraction.

    Returns:
        vtk.vtkPolyData: A new vtkPolyData containing only the specified label.
    """
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(msh)
    threshold.ThresholdBetween(label, label)  # Keep only the selected label
    threshold.SetInputArrayToProcess(0, 0, 0, 1, scalar_field)  # Ensure correct scalar selection
    threshold.Update()

    # Convert vtkUnstructuredGrid to vtkPolyData
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(threshold.GetOutput())
    geo_filter.Update()

    return geo_filter.GetOutput()

def clean_mesh(msh: vtk.vtkPolyData) : 
    """
    Clean a vtkPolyData object by removing duplicate points and cells.
    """
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputData(msh)
    cleanFilter.Update()

    return cleanFilter.GetOutput()

# def detect_bridges_combined(polydata: vtk.vtkPolyData, max_distance=5.0, thickness_threshold=1.5) -> vtk.vtkIntArray:
#     """
#     Combine graph-based and thickness-based methods. A cell is flagged only if both
#     methods flag it as a bridge.
#     """
#     # Compute the individual flags.
#     graphFlag = detect_bridges_with_graph(polydata)
#     thicknessFlag = detect_bridges_with_thickness(polydata, max_distance, thickness_threshold)
    
#     numCells = polydata.GetNumberOfCells()
#     combinedFlag = vtk.vtkIntArray()
#     combinedFlag.SetName("CombinedBridgeFlag")
#     combinedFlag.SetNumberOfComponents(1)
#     combinedFlag.SetNumberOfTuples(numCells)
    
#     for i in range(numCells):
#         # Logical AND: flag cell if both methods flag it.
#         if graphFlag.GetTuple1(i) == 1 and thicknessFlag.GetTuple1(i) == 1:
#             combinedFlag.SetTuple1(i, 1)
#         else:
#             combinedFlag.SetTuple1(i, 0)
    
#     return combinedFlag
def compute_mesh_size(msh) -> tuple:
    """
    Compute the size of a mesh by calculating the sum of the areas of all cells.
    """
    total_area = 0.0
    for i in range(msh.GetNumberOfCells()):
        cell = msh.GetCell(i)
        total_area += cell.ComputeArea()
    
    return msh.GetNumberOfCells(), total_area