# README

## Quick guide - install  (macOS/linux) 

Download this repository using git
```
mkdir -p ~/code & git clone https://github.com/alonsoJASL/imatools ~/code/imatools
```

**Setup conda environment**
> You only need to do this once

Download [anaconda](https://www.anaconda.com/products/distribution), then 
on a terminal type: 
```
conda create -n imatools python=3.8 -y & conda activate imatools
```

Copy the following to install the python dependencies of this project
```
conda install -c conda-forge vtk itk numpy scipy matplotlib pandas seaborn networkx scikit-image -n imatools -y
```

## Example: Calculate volume & area of vtk file
Open a Terminal. Navigate to the code folder:
```
cd ~/code/imatools
```
Activate the anaconda environment
```
conda activate imatools
```
Run the volume code by typing 
```
python calculate_volume.py /path/to/file.vtk
``` 
> You can type `python calculate_volume.py` + <kbd>SPACE</kbd> and then drag the file from Finder into the terminal window if you do not know the path.

![SCR-20220905-khk-2](https://user-images.githubusercontent.com/9891700/188464906-970f6098-064a-48e1-a138-19e4ba43715b.jpeg)


## Alternative minimal setup
To use with VTK 8.1 to save vtk meshes in legacy VTK Writer 4.2.

```
conda create -n vtk81 python=3.6 -y
conda activate vtk81
conda install -c conda-forge vtk=8.1 -y
conda install -c conda-forge numpy -n vtk81 -y  
conda install -c conda-forge itk
```
