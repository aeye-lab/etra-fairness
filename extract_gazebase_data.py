import argparse
import glob
import os
import zipfile


def extract_gazebase_data(data_dir: str = '.') -> int:
    for _file in glob.glob(f'{data_dir}/*', recursive=True):
        if os.path.exists(_file.split('.zip')[0]):
            continue
        if _file.endswith('.zip'):
            with zipfile.ZipFile(_file, 'r') as zip:
                zip.extractall(path=f"./{_file.split('.zip')[0]}")
                print(f' Extracted file {_file}', end='\r')
        else:
            continue

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', type=str, default='.', help='Specify Gazebase path %s(deault)s',
    )
    args = parser.parse_args()
    print('Start extracting gazebase main folder...')
    extract_gazebase_data(args.data_dir)
    print('Finished extracting gazebase main folder...')

    print('Start extracting subfolder...')
    extract_gazebase_data(data_dir='GazeBase_v2_0/Round*')
    print('\nExtraction done -- you can now start experimenting!')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
