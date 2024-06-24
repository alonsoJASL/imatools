# README

## Quick guide - install  (macOS/linux) 

Download this repository using git
```
mkdir -p ~/code & git clone https://github.com/alonsoJASL/imatools ~/code/imatools
```

## Environments
### Option 1 (Recommended). Use Conda + Poetry
[Poetry](https://python-poetry.org/docs/) is a tool for dependency management. 
We create the environment with Anaconda and manage dependencies with Poetry.

1. [Install Poetry](https://python-poetry.org/docs/) in your computer
2. Install Anaconda
3. Create a new environment with the file provided: `conda env create -f environment.yaml` 
4. Activate the environment `conda activate imatools` 
5. Install the dependencies using Poetry `poetry install --only main`.

Poetry will read from the poetry.lock file and install the exact dependencies used to develop this project.

> If you want to experiment with dependencies versions, delete the lock file and modify the pyproject.toml before `poetry install --only main`  

### Option 2. Only use CONDA
Anaconda is useful, but it can be slow at managing dependencies. It is recommended to use poetry.

**Setup conda environment**
> You only need to do this once

Download [anaconda](https://www.anaconda.com/products/distribution), then 
on a terminal type: 
```
conda create -n imatools python=3.9 -y & conda activate imatools
```

Copy the following to install the python dependencies of this project
```
conda install -c conda-forge vtk=9.2.6 simpleitk numpy scipy=1.9.2 matplotlib pandas seaborn networkx scikit-image nibabel pydicom -n imatools -y
```

### Adjust `PYTHONPATH` 
Sometimes you will need to set the `PYTHONPATH` variable, do this by opening a termina, changing into imatools directory `cd /path/to/imatools` and pasting the following: 
```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)
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


## Some minimal tutorials: 
+ [Create surface meshes from your multilabel segmentations](https://hackmd.io/@jsolislemus/HkB3yR8Ka)

## Alternative minimal setup#
> While this is OK, it is not necessary as we use VTK 9.2.6, which
> allows to specify the version of Legacy VTK ASCII files.
To use with VTK 8.1 to save vtk meshes in legacy VTK Writer 4.2.

```
conda create -n vtk81 python=3.6 -y
conda activate vtk81
conda install -c conda-forge vtk=8.1 -y
conda install -c conda-forge numpy -n vtk81 -y  
conda install -c conda-forge itk
```
