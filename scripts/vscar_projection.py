import os 
import argparse 

from imatools.common.CommandRunner import CommandRunner
from imatools.common.config import configure_logging, add_file_handler


def parse_input_name(input_path) :
    base = os.path.basename(input_path)
    dirname = os.path.dirname(input_path)
    name, ext = os.path.splitext(base)

    return {'base': base, 'dirname': dirname, 'name': name, 'ext': ext} 

def main(args) : 
    print(args)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Project Ventricular Scar from CINE ")
    parser.add_argument('mode', type=str, choices=['msh2mm', 'deform', 'cog', 'scar'], help='Mode of operation')
    parser.add_argument('--input', '-in', type=str, required=True, help='Input file')
    # mode      | input 
    # msh2mm    | msh_cine 
    # deform    | msh_cine_mm 
    # cog       | msh_cine_mm_on_LGE
    # scar      | scar_cine_mm_on_LGE

    deform_group = parser.add_argument_group('Deform Options')
    deform_group.add_argument('--path-to-mirtk', '-mirtk', type=str, default=None, help='Folder to MIRTK executables')
    deform_group.add_argument('--path-to-moving', '-moving', type=str, default=None, help='Path to moving image (CINE)')
    deform_group.add_argument('--path-to-fixed', '-fixed', type=str, default=None, help='Path to fixed image (LGE)')

    cog_group = parser.add_argument_group('Center of Gravity Options')
    

    