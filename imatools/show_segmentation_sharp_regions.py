import os 
import argparse 
import vtk
import math

import numpy as np

from vtk.util import numpy_support 
from common import itktools as itku

NEIGHBOUR_SCALARS='NeighborhoodSharpness'

def get_vtk_reader(input_path) : 
    _, ext = os.path.splitext(input_path)
    if ext == '.nii' or ext == '.nii.gz' : 
        reader = vtk.vtkNIFTIImageReader()
    elif ext == '.mha' or ext == '.mhd' : 
        reader = vtk.vtkMetaImageReader()
    elif ext == '.nrrd' :
        reader = vtk.vtkNrrdReader()
    else : 
        raise ValueError(f'Unknown extension: {ext}')
    
    reader.SetFileName(input_path)
    reader.Update()
    
    return reader

def custom_lut_hot(neighborhoodArray) : 
    # --- Create a custom "hot" lookup table ---
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetTableRange(neighborhoodArray.GetRange())
    lut.Build()
    for i in range(256):
        t = i / 255.0
        # Define the "hot" colormap:
        # First third: black to red
        if t < 1/3.0:
             r = 3.0 * t
             g = 0.0
             b = 0.0
        # Second third: red to yellow
        elif t < 2/3.0:
             r = 1.0
             g = 3.0 * t - 1.0
             b = 0.0
        # Final third: yellow to white
        else:
             r = 1.0
             g = 1.0
             b = 3.0 * t - 2.0
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.Build()
    return lut


def custom_lut_jet(neighborhoodArray):
    """Create a lookup table with colors similar to macOS Preview's default colormap."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetTableRange(neighborhoodArray.GetRange())
    lut.Build()

    for i in range(256):
        t = i / 255.0  # Normalize between 0 and 1
        
        # Define the "Preview" colormap:
        # Blue → Green → Yellow → Red
        if t < 0.25:  # Blue to Cyan
            r = 0.0
            g = 4.0 * t
            b = 1.0
        elif t < 0.5:  # Cyan to Green
            r = 0.0
            g = 1.0
            b = 2.0 - 4.0 * t
        elif t < 0.75:  # Green to Yellow
            r = 4.0 * (t - 0.5)
            g = 1.0
            b = 0.0
        else:  # Yellow to Red
            r = 1.0
            g = 1.0 - 4.0 * (t - 0.75)
            b = 0.0
        
        lut.SetTableValue(i, r, g, b, 1.0)  # Alpha = 1.0 (fully opaque)

    lut.Build()
    return lut

def paraview_lut(neighborhoodArray):
    """Create a lookup table with colors matching ParaView's default 'Cool to Warm' colormap."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetTableRange(neighborhoodArray.GetRange())
    lut.Build()

    for i in range(256):
        t = i / 255.0  # Normalize between 0 and 1

        # ParaView "Cool to Warm" color transitions:
        # Blue → White → Red
        if t < 0.5:  # Blue to White
            r = 2.0 * t
            g = 2.0 * t
            b = 1.0
        else:  # White to Red
            r = 1.0
            g = 2.0 - 2.0 * t
            b = 2.0 - 2.0 * t

        lut.SetTableValue(i, r, g, b, 1.0)  # Alpha = 1.0 (fully opaque)

    lut.Build()
    return lut

def highlight_neighbours_default(originalPolyData: vtk.vtkPolyData, smoothedPolyData: vtk.vtkPolyData, neighbourhood_radius) -> vtk.vtkDoubleArray: 
    # Build a point locator (KD-tree) for the original mesh.
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(originalPolyData)
    pointLocator.BuildLocator()

    # Create an array to store the neighborhood modifications.
    neighborhoodArray = vtk.vtkDoubleArray()
    neighborhoodArray.SetNumberOfComponents(1)
    neighborhoodArray.SetNumberOfTuples(originalPolyData.GetNumberOfPoints())
    neighborhoodArray.FillComponent(0, 0.0)

    neighborhoodArray.SetName(NEIGHBOUR_SCALARS)

    # Define a radius for neighbor search. Adjust as appropriate for your data.
    neighborRadius = neighbourhood_radius

    numSmoothPoints = smoothedPolyData.GetNumberOfPoints()
    for i in range(numSmoothPoints):
        # Get the point coordinates from the smoother output.
        point = smoothedPolyData.GetPoint(i)
        # Find the closest point in the original mesh.
        closestId = pointLocator.FindClosestPoint(point)
        # Add 1.0 to the closest point.
        currentVal = neighborhoodArray.GetTuple1(closestId)
        neighborhoodArray.SetTuple1(closestId, currentVal + 1.0)
        
        # Find immediate neighbors within the specified radius.
        neighborIds = vtk.vtkIdList()
        pointLocator.FindPointsWithinRadius(neighborRadius, point, neighborIds)
        for j in range(neighborIds.GetNumberOfIds()):
            neighborId = neighborIds.GetId(j)
            # Optionally skip the central point if you don't want to add extra to it.
            if neighborId == closestId:
                continue
            neighborVal = neighborhoodArray.GetTuple1(neighborId)
            neighborhoodArray.SetTuple1(neighborId, neighborVal + 0.5)
    
    return neighborhoodArray

def highlight_neighbours_decreasing(originalPolyData: vtk.vtkPolyData,  smoothedPolyData: vtk.vtkPolyData,  neighbourhood_radius) -> vtk.vtkDoubleArray:
    """
    This function highlights points in the originalPolyData based on the locations in the smoothedPolyData.
    It assigns a full weight (1.0) to the closest point and a decreasing weight (up to 0.5 at distance 0)
    to its neighbors, where the weight decreases linearly with distance.
    
    Parameters:
      originalPolyData: vtkPolyData representing the original mesh.
      smoothedPolyData: vtkPolyData representing the smoothed sharp regions.
      neighbourhood_radius: float value defining the search radius for neighbor points.
    
    Returns:
      A vtkDoubleArray with the computed "neighbourhood" values.
    """
    # Build a point locator (KD-tree) for the original mesh.
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(originalPolyData)
    pointLocator.BuildLocator()

    # Create an array to store the neighborhood modifications.
    neighborhoodArray = vtk.vtkDoubleArray()
    neighborhoodArray.SetNumberOfComponents(1)
    neighborhoodArray.SetNumberOfTuples(originalPolyData.GetNumberOfPoints())
    neighborhoodArray.FillComponent(0, 0.0)
    
    # Define a name for the scalar array (you can change this as needed).
    neighborhoodArray.SetName(NEIGHBOUR_SCALARS)

    neighborRadius = neighbourhood_radius

    numSmoothPoints = smoothedPolyData.GetNumberOfPoints()
    for i in range(numSmoothPoints):
        # Get the point coordinates from the smoothed output.
        point = smoothedPolyData.GetPoint(i)
        # Find the closest point in the original mesh.
        closestId = pointLocator.FindClosestPoint(point)
        # Add 1.0 to the closest point.
        currentVal = neighborhoodArray.GetTuple1(closestId)
        neighborhoodArray.SetTuple1(closestId, currentVal + 1.0)
        
        # Find immediate neighbors within the specified radius.
        neighborIds = vtk.vtkIdList()
        pointLocator.FindPointsWithinRadius(neighborRadius, point, neighborIds)
        for j in range(neighborIds.GetNumberOfIds()):
            neighborId = neighborIds.GetId(j)
            # Optionally skip the central point if you don't want to add extra to it.
            if neighborId == closestId:
                continue
            neighborCoord = originalPolyData.GetPoint(neighborId)
            # Compute the Euclidean distance between the smoothed point and the neighbor.
            d = math.sqrt((point[0] - neighborCoord[0])**2 +
                          (point[1] - neighborCoord[1])**2 +
                          (point[2] - neighborCoord[2])**2)
            # Compute a weight that decreases linearly with distance.
            weight = 0.5 * (1 - d / neighborRadius)
            # Clamp the weight to 0 if it's negative.
            if weight < 0:
                weight = 0.0
            neighborVal = neighborhoodArray.GetTuple1(neighborId)
            neighborhoodArray.SetTuple1(neighborId, neighborVal + weight)
    
    return neighborhoodArray

def marching_cubes(im, label=None) -> vtk.vtkMarchingCubes: 
    if label is not None : 
        im = itku.extract_single_label(im, label, binarise=True)
    spacing = im.GetSpacing()  # assuming itku.load_image returns an ITK image
    im_array = itku.imview(im)
    vtk_image = vtk.vtkImageData()
    dims = im_array.shape
    vtk_image.SetDimensions(dims[2], dims[1], dims[0])
    vtk_image.SetSpacing(spacing)
    # vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)  

    vtk_array = numpy_support.numpy_to_vtk(num_array=im_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_image.GetPointData().SetScalars(vtk_array)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_image)
    # mc.SetInputConnection(vtk_image.GetOutputPort())
    mc.SetValue(0, 1)  # Assumes segmentation label is 1 (adjust if needed)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.ComputeScalarsOn()
    mc.Update()

    return mc    

def main(args) : 
    # -------------------------
    print('Step 1: Load the segmentation image')
    # -------------------------
    # reader = get_vtk_reader(args.input)
    # Replace with your segmentation file (e.g., NIfTI, MetaImage, etc.)
    seg = itku.load_image(args.input)
    
    print('Step 2: Marching Cubes to extract the surface mesh')
    mc = marching_cubes(seg, args.label)

    # -------------------------
    print('Step 3: Compute curvature on the mesh')
    # -------------------------
    curvatureFilter = vtk.vtkCurvatures()
    curvatureFilter.SetInputConnection(mc.GetOutputPort())
    if args.curvature_type == 'mean' :
        curvatureFilter.SetCurvatureTypeToMean()
    elif args.curvature_type == 'gaussian' :
        curvatureFilter.SetCurvatureTypeToGaussian()
    elif args.curvature_type == 'maximum' :
        curvatureFilter.SetCurvatureTypeToMaximum()
    else :
        raise ValueError(f'Unknown curvature type: {args.curvature_type}')
    curvatureFilter.Update()

    print("Curvature Scalar Range:", curvatureFilter.GetOutput().GetPointData().GetScalars().GetRange())

    # -------------------------
    print('Step 4: Threshold high-curvature (sharp) regions')
    # -------------------------
    # High curvature values indicate areas that are “pointy.”
    # Adjust the threshold value based on your dataset.
    thresholdFilter = vtk.vtkThreshold()
    thresholdFilter.SetInputConnection(curvatureFilter.GetOutputPort())
    thresholdFilter.SetLowerThreshold(args.curvature_sensitivity)  # Only pass regions with curvature > 5.0
    thresholdFilter.Update()

    # The output of vtkThreshold is an unstructured grid.
    # Convert it back to polydata for further processing.
    geoFilter = vtk.vtkGeometryFilter()
    geoFilter.SetInputConnection(thresholdFilter.GetOutputPort())
    geoFilter.Update()

    # -------------------------
    print('Step 5: Smooth the detected sharp regions')
    # -------------------------
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(geoFilter.GetOutputPort())
    smoother.SetNumberOfIterations(30)  # Increase iterations for more smoothing (20, 10)
    smoother.SetPassBand(0.1)           # Controls the smoothing strength; adjust as needed (0.1, 0.05)
    smoother.BoundarySmoothingOff()     # Typically turn off boundary smoothing
    smoother.Update()
    
    # -------------------------
    print('Step 6: Neighborhood modification based on smoothed sharp points')
    # -------------------------
    # We'll modify the original mesh by adding a scalar value:
    originalPolyData = mc.GetOutput()
    smoothedPolyData = smoother.GetOutput()

    # neighborhoodArray = highlight_neighbours_default(originalPolyData, smoothedPolyData, args.neighbourhood_radius)
    neighborhoodArray = highlight_neighbours_decreasing(originalPolyData, smoothedPolyData, args.neighbourhood_radius)
    
    # Add the new neighborhood array to the original mesh.
    originalPolyData.GetPointData().AddArray(neighborhoodArray)
    originalPolyData.GetPointData().SetActiveScalars(NEIGHBOUR_SCALARS)

    # -------------------------
    print('Step 7: Visualization of the original and smoothed regions')
    # -------------------------

    # Mapper and actor for the original segmentation mesh (semi-transparent for reference)
    mapperOrig = vtk.vtkPolyDataMapper()
    mapperOrig.SetInputConnection(mc.GetOutputPort())
    mapperOrig.ScalarVisibilityOff()

    actorOrig = vtk.vtkActor()
    actorOrig.SetMapper(mapperOrig)
    actorOrig.GetProperty().SetOpacity(0.3)  # Make it translucent
    

    # Mapper and actor for the smoothed sharp regions (displayed in red)
    mapperSmooth = vtk.vtkPolyDataMapper()
    mapperSmooth.SetInputConnection(smoother.GetOutputPort())
    mapperSmooth.ScalarVisibilityOff()

    actorSmooth = vtk.vtkActor()
    actorSmooth.SetMapper(mapperSmooth)
    actorSmooth.GetProperty().SetColor(1, 0, 0)  # Red color for the smoothed regions
    
    # Create a mapper and actor for visualizing the neighborhood modifications.
    mapperNeighborhood = vtk.vtkPolyDataMapper()
    mapperNeighborhood.SetInputData(originalPolyData)
    mapperNeighborhood.SetScalarVisibility(True)
    mapperNeighborhood.SetScalarRange(neighborhoodArray.GetRange())

    mapperNeighborhood.SetLookupTable(paraview_lut(neighborhoodArray)) # Assign the lookup table to the mapper.
    mapperNeighborhood.SetUseLookupTableScalarRange(True)

    actorNeighborhood = vtk.vtkActor()
    actorNeighborhood.SetMapper(mapperNeighborhood)
    # actorNeighborhood.GetProperty().SetOpacity(0.95)  # Make it translucent

    light = vtk.vtkLight()
    # light.SetPosition(1, 1, 1)
    light.SetLightTypeToCameraLight()

    # Create renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderer.AddLight(light)

    if args.show_heart :
        print('Step 8: Adding visuatisation of the full heart')
        mc_full = marching_cubes(itku.remove_label(seg, args.label))

        mapperFull = vtk.vtkPolyDataMapper()
        mapperFull.SetInputConnection(mc_full.GetOutputPort())
        mapperFull.ScalarVisibilityOff()

        actorFull = vtk.vtkActor()
        actorFull.SetMapper(mapperFull)
        actorFull.GetProperty().SetOpacity(0.1)  # Make it translucent
        renderer.AddActor(actorFull)

    if args.expand_sharp_regions :
        renderer.AddActor(actorNeighborhood)
    else :
        renderer.AddActor(actorOrig)
        renderer.AddActor(actorSmooth)
    renderer.SetBackground(0.1, 0.1, 0.1)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.SetRenderWindow(renderWindow)

    # Render and start interaction
    renderWindow.Render()
    interactor.Start()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Detect and visualize sharp regions in a segmentation')
    parser.add_argument('--input', '-in', required=True, help='Input segmentation image')
    parser.add_argument('--label', '-l', required=True, help='Label value for the segmentation', type=int)
    parser.add_argument('--show-heart', '-show-heart', action='store_true', help='Show the full heart')

    curvature_group = parser.add_argument_group('Curvature options')
    curvature_group.add_argument('--curvature-type', '-curvature-type', default='mean', choices=['mean', 'gaussian', 'maximum'], help='Curvature type')
    curvature_group.add_argument('--curvature-sensitivity', '-curvature-sensitivity', type=float, default=5.0, help='Curvature sensitivity')
    
    sharp_regions_group = parser.add_argument_group('Sharp regions options')
    sharp_regions_group.add_argument('--expand-sharp-regions', '-expand-sharp-regions', action='store_true', help='Expand sharp regions')
    sharp_regions_group.add_argument('--neighbourhood-radius', '-neighbourhood-radius', type=float, default=1.0, help='Neighborhood radius')
    args = parser.parse_args()
    main(args)
