# README

Setup conda environment
```
conda create -n imatools python=3.8 -y
conda activate imatools
```

installs
```
conda install -c conda-forge vtk itk numpy scipy matplotlib pandas seaborn networkx -n imatools -y
```


## Alternative minimal setup
To use with VTK 8.1 to save vtk meshes in legacy VTK Writer 4.2.

```
conda create -n vtk81 python=3.6 -y
conda activate vtk81
conda install -c conda-forge vtk=8.1 -y
conda install -c conda-forge numpy -n vtk81 -y  
```
