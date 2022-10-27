import sys
import vtk
import vtk.util.numpy_support as vtknp
import numpy as np

def readVtk(fname):
    """
    Read VTK file
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()

    return reader.GetOutput()

def writeVtk(mesh, directory, outname="output"):
    writer=vtk.vtkPolyDataWriter()
    writer.WriteArrayMetaDataOff()
    writer.SetInputData(mesh)
    writer.SetFileName(directory+"/"+outname+".vtk")
    writer.SetFileTypeToASCII()
    writer.Update()

def setCellDataToPointData(msh, fieldname='scalars') : 
    c2pt = vtk.vtkCellDataToPointData()
    c2pt.SetInputData(msh)
    c2pt.PassCellDataOn()
    c2pt.SetContributingCellOption(0)
    c2pt.Update()

    omsh=c2pt.GetPolyDataOutput()
    omsh.GetPointData().GetScalars().SetName(fieldname)

    return omsh

def getCentreOfGravity(msh) : 
    pts, el = extractPointsAndElemsFromVtk(msh)
    cog = np.zeros((len(el), 3))
    for ix in range(len(el)): 
        ex = el[ix] 
        for jx in range(3) : 
            cog[ix,:] += pts[ex[jx]] 
        cog[ix, :] /= 3
    
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
    cog = getCentreOfGravity(msh_source)
    for ix in range(msh_source.GetNumberOfCells()):
        pt = cog[ix, :]
        closest_pt = np.zeros((3,1))
        c_id=np.int8()
        subid = np.int8()
        dist2 = np.float32()
        id_on_target=vtk.reference(c_id)
        Subid=vtk.reference(subid)
        Dist2=vtk.reference(dist2)

        # target_pl.FindCell(pt)
        target_pl.FindClosestPoint(pt, closest_pt, id_on_target, Subid, Dist2)

        mapped_val = target_scalar.GetTuple1(id_on_target) if (
            id_on_target > 0) else default_value
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

def fibrosisOverlapCell(msh0, msh1, th0, th1=None, name0='msh0', name1='msh1') : 
    """Make sure msh0 aligns with msh1 in number of cells"""
    th1 = th0 if (th1 is None) else th1

    omsh = vtk.vtkPolyData()
    omsh.DeepCopy(msh0)
    o_scalar = vtk.vtkFloatArray()
    o_scalar.SetNumberOfComponents(1)
    
    scalar0 = msh0.GetCellData().GetScalars()
    scalar1 = msh1.GetCellData().GetScalars()

    count0 = 0
    count1 = 0
    countb = 0
    countt = msh0.GetNumberOfCells()

    for ix in range(msh0.GetNumberOfCells()):
        value_assigned = 0

        if (scalar0.GetTuple1(ix) == 0):
            value_assigned = -1
            countt -= 1

        else:
            fib_at_0 = False
            fib_at_1 = False
            if (scalar0.GetTuple1(ix) >= th0):
                value_assigned += 1
                count0 += 1
                fib_at_0 = True

            if (scalar1.GetTuple1(ix) >= th1):
                value_assigned += 2
                count1 += 1
                fib_at_1 = True 
            
            countb += 1 if (fib_at_1 and fib_at_0) else 0

        o_scalar.InsertNextTuple1(value_assigned)

    omsh.GetCellData().SetScalars(o_scalar)
    tn = countt - (count0+count1+countb)
    count_dic = {'total' : countt, name0 : count0, name1: count1, 'overlap' : countb, 'none' : tn}
    return omsh, count_dic

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