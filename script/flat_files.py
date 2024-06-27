import argparse
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(description='Flatten files in structured directories in to the destination directory.')
    parser.add_argument('src', help='source directory')
    parser.add_argument('dst', help='destination directory')
    parser.add_argument('--ext', help='file extension to flatten', default='jpg')
    parser.add_argument('--delimiter', help='delimiter for flattend file name', default='_')


    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    ext = args.ext
    delimiter = args.delimiter

    if not src.is_dir():
        raise ValueError(f'{src} is not a directory')
    
    if not dst.is_dir():
        raise ValueError(f'{dst} is not a directory')
    
    for p in src.glob(f'**/*.{ext}'):
        rel = p.relative_to(src)
        name = delimiter.join(rel.parts)
        dst_file = dst / name
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst_file)

if __name__ == '__main__':
    main()



