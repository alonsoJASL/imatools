import argparse
import os
import common.vtktools as vtku 

def save_array(array, filename, is_elem=False) : 
    with open(filename, 'w') as f : 
        f.write(f'{len(array)}\n')
        for a in array : 
            if is_elem : 
                f.write(f'Tt {a[0]} {a[1]} {a[2]} 0\n')
            else :
                f.write(f'{a[0]} {a[1]} {a[2]}\n')

def main(args) : 
    folder = os.path.dirname(args.input)
    name = os.path.basename(args.input)
    if args.output == '' : 
        args.output = name.replace('.mesh', '')
    
    gen_attr, pts_attr, elem_attr = vtku.parse_dotmesh_file(args.input, 'iso-8859-1')
    pts = pts_attr['points']
    elem = elem_attr['elements']

    save_array(pts, f'{os.path.join(folder, args.output)}.pts')
    save_array(elem, f'{os.path.join(folder, args.output)}.elem', True)

    print(f'VTK file saved as {args.output}')

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description="Convert dotmesh to carp")
    parser.add_argument("-in", "--input", help="Input dotmesh file")
    parser.add_argument("-out", "--output", required=False, default='', help="Output name")
    
    args = parser.parse_args()
    main(args)