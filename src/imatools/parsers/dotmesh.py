# src/imatools/parsers/dotmesh.py
"""Parsers for Biosense Webster .mesh files and CARP-style text arrays.

Functions migrated from:
- ``imatools.common.vtktools`` — ``parse_dotmesh_file``
- ``imatools.convert_dotmesh``  — ``save_array``

The old import paths still resolve via bottom-of-file shims in the source
modules (T2b4).
"""

from __future__ import annotations

import re

import numpy as np


def parse_dotmesh_file(file_path, myencoding="utf-8"):
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
    with open(file_path, "r", encoding=myencoding) as file:
        mesh_data = file.readlines()

    # Initialize dictionaries to store parsed data
    accepatable_keys = ["MeshID", "MeshName", "NumVertex", "NumTriangle"]
    comments = ["#", ";", "//"]
    general_attributes = {"MeshID": None, "MeshName": "not-set", "NumVertex": 0, "NumTriangle": 0}
    vertices_section = {}
    triangles_section = {}

    current_section = None

    # Define regular expressions to match section headers
    general_attr_regex = re.compile(r"\[GeneralAttributes\]")
    vertices_section_regex = re.compile(r"\[VerticesSection\]")
    triangles_section_regex = re.compile(r"\[TrianglesSection\]")

    for line in mesh_data:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        if line[0] in comments:
            continue
        if general_attr_regex.match(line):
            current_section = "GeneralAttributes"
            continue

        if "=" in line:
            key, value = line.split("=")
            if key.strip() in accepatable_keys:
                general_attributes[key.strip()] = value.strip()

    print(general_attributes)
    n_vertex = int(general_attributes["NumVertex"])
    n_triangle = int(general_attributes["NumTriangle"])

    vertices_section = {
        "index": np.ndarray(n_vertex, dtype=np.uint16),
        "points": np.ndarray((n_vertex, 3), dtype=float),
    }
    triangles_section = {
        "index": np.ndarray(n_triangle, dtype=np.uint16),
        "elements": np.ndarray((n_triangle, 3), dtype=np.uint16),
    }

    count_vertices = 0
    count_triangles = 0
    for line in mesh_data:
        if not line:
            continue  # Skip empty lines
        if line[0] in comments:
            continue
        if vertices_section_regex.match(line):
            current_section = "VerticesSection"
            continue

        if triangles_section_regex.match(line):
            current_section = "TrianglesSection"
            continue

        if current_section == "VerticesSection":
            if "=" in line:
                index, data = line.strip().split("=")
                vertices_section["index"][count_vertices] = np.uint16(index.strip())
                vertices_section["points"][count_vertices, :] = [
                    float(x) for x in data.strip().split()[:3]
                ]
                count_vertices += 1
                if count_vertices == n_vertex:
                    continue

        if current_section == "TrianglesSection":
            if "=" in line:
                index, data = line.strip().split("=")
                triangles_section["index"][count_triangles] = np.uint16(index.strip())
                triangles_section["elements"][count_triangles, :] = [
                    np.uint16(x) for x in data.strip().split()[:3]
                ]
                count_triangles += 1
                if count_triangles == n_triangle:
                    current_section = None
                    continue

    return general_attributes, vertices_section, triangles_section


def save_array(array, filename, is_elem=False):
    with open(filename, "w") as f:
        f.write(f"{len(array)}\n")
        for a in array:
            if is_elem:
                f.write(f"Tt {a[0]} {a[1]} {a[2]} 0\n")
            else:
                f.write(f"{a[0]} {a[1]} {a[2]}\n")
