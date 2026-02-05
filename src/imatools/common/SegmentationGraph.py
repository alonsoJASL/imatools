import networkx as nx
import numpy as np
import SimpleITK as sitk
from itertools import product
from scipy.sparse.linalg import eigsh
from common import itktools as itku

class SegmentationGraph:
    def __init__(self, im: sitk.Image):
        self.image = im
        self.graph = nx.Graph()
        self._create_graph()

    def _create_graph(self):
        array = itku.imview(self.image)
        for index in np.ndindex(array.shape):
            self.graph.add_node(index)
            for offset in product([-1, 0, 1], repeat=array.ndim):
                neighbor_index = tuple(i + o for i, o in zip(index, offset))
                if self._is_valid_index(neighbor_index, array.shape):
                    self.graph.add_edge(index, neighbor_index)

    def _is_valid_index(self, index, shape):
        return all(0 <= i < s for i, s in zip(index, shape))

    def get_graph(self):
        return self.graph
    
    def num_nodes(self) : 
        return self.graph.number_of_nodes()
        
    def laplacian(self) : 
        return nx.laplacian_matrix(self.graph)

    def eigen(self, k=-1, which_ord='LM') : 
        lap_mat = self.laplacian()
        k = self.num_nodes() if k==-1 else k

        evals, evects = eigsh(lap_mat.asftype(), k, which=which_ord)
        return evals, evects
    
    def smooth(self, k: int):
        """Smooth the segmentation by retaining only the eigenvectors corresponding to the smallest k eigenvalues."""
        eigenvalues, eigenvectors = self.eigen(k, which_ord='SM')
        smooth_array = np.dot(eigenvectors, eigenvectors.T)
        smooth_image = itku.arrayview(smooth_array, self.image)
        return smooth_image