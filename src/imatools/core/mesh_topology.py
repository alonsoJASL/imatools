# src/imatools/core/mesh_topology.py
"""Graph-based mesh connectivity analysis migrated from
``imatools.common.vtktools`` (M2a-2; zero-caller-but-KEEP functions relocated
per Jose's M2 review — see MIGRATION_M2.md).

Bridge/thin-region detection over a triangle-mesh cell-adjacency graph, built
on ``networkx``. The 5 functions here are the authoritative implementations;
``imatools.common.vtktools`` no longer holds them (full removal, no shim —
they had zero callers anywhere).

All functions are self-contained (``networkx``/``numpy``/``vtk`` only, no
cross-module calls), so no lazy accessor is needed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import vtk


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
    edge2cells = (
        {}
    )  # Maps edges (sorted tuples of point IDs) to the list of cell IDs that share them.
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
            tuple(sorted((ptIds[2], ptIds[0]))),
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
    for cellA, cellB in bridges:
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
            tuple(sorted((ptIds[2], ptIds[0]))),
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
    for cellA, cellB in bridges:
        bridgeFlag.SetTuple1(cellA, 1)
        bridgeFlag.SetTuple1(cellB, 1)

    return bridgeFlag


def detect_bridges_with_thickness(
    polydata: vtk.vtkPolyData, max_distance=5.0, thickness_threshold=1.5, output_raw_thickness=True
) -> vtk.vtkDoubleArray:
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
        end = (
            point[0] - normal[0] * max_distance,
            point[1] - normal[1] * max_distance,
            point[2] - normal[2] * max_distance,
        )
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
