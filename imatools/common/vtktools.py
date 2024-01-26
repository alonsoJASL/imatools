import sys
import os

import numpy as np
import pandas as pd
import vtk
import vtk.util.numpy_support as vtknp

from imatools.common.config import configure_logging 

logger = configure_logging(__name__)

def l2_norm(a): return np.linalg.norm(a, axis=1)
def dot_prod_vec(a,b): return np.sum(a*b, axis=1)

def readVtk(fname):
    """
    Read VTK file
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()

    return reader.GetOutput()

def writeVtk(mesh, directory, outname="output"):
    filename = os.path.join(directory, outname)
    filename += ".vtk" if not filename.endswith(".vtk") else ""

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

def get_cog_per_element(msh):
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