"""
@Purpose: Bulk import packages into Anki.

@Usage:   Can be run w/ no configuration on first run through.

          If you've made any changes to previoulsy imported cards in Anki,
          including adding audio via HyperTTS, then re-importing
          from packages will overwrite those changes. This is Anki behavior.

          START is provided as a workaround so that you can bulk import only
          packages newer than your given file start index.

          See Easy_Run_s1_s7.py config section for more on START.

"""
import math
import os
import platform
import time
import re

from anki.errors import DBError
from anki.collection import ImportAnkiPackageRequest, Collection

from s5_Generate_Anki_Package import PACKAGE_DIR, CHUNK_SIZE

#======== Configuraton ===========
PROFILE = 'User 1'  # default is 'User 1'; see Anki->File->Switch Profile to see what you have. this is case sensitive
START = 1           # see @Instructions or Easy_Run_s1_s7.py for info.
PACKAGE_APKG = 'apkg'
PACKAGE_COLAPKG = 'colpkg'
USER_PATH = os.path.expanduser('~')
#=================================


def main():
    import_anki_packages()
    return


def import_anki_packages(start=START):
    print('Importing packages into Anki...')

    # get list of packages
    packages = get_packages(start)

    # open collection
    collection_anki2_path = get_collection_path(PROFILE)
    print(f'Found {collection_anki2_path}.')
    try:
        collection = Collection(collection_anki2_path)

        # bulk import packages in Anki
        print('Importing anki packages...')
        for package in packages:
            print(f'  Importing package {package}.')
            package_path = f'{PACKAGE_DIR}/{package}'
            collection.import_anki_package(
                ImportAnkiPackageRequest(
                    package_path=package_path,
                )
            )
        print()
        print('Success. Imported anki packages.')
    except DBError:
        print('Error importing anki packages. Ensure Anki is closed. Dumping crash log...')
        time.sleep(3)
        raise

    print('Imported packages.')
    print()
    return


def get_packages(start):
    def parse_package_id(pkg_filename):
        matches = list(re.finditer(r'_(\d+)', pkg_filename))
        if matches:
            return int(matches[-1].group(1))
        return None

    dir_contents = os.listdir(PACKAGE_DIR)
    import_packages_starting_with_pkgid = math.ceil(start / CHUNK_SIZE)

    packages = []
    for package_name in dir_contents:
        # select packages
        if (os.path.isfile(os.path.join(PACKAGE_DIR, package_name)) and
                (PACKAGE_APKG in package_name.lower() or PACKAGE_COLAPKG in package_name.lower())):

            # select packages not previously imported
            package_num = parse_package_id(package_name)
            if package_num >= import_packages_starting_with_pkgid:
                packages.append(package_name)
    print('Found and read packages.')
    return packages


def get_collection_path(profile_name, attempts=0):
    def get_anki_dir(attempts, *anki_dir_tuple):
        anki_dir_tuple = anki_dir_tuple[attempts:]  # try progressively older locations
        for path in anki_dir_tuple:
            if os.path.exists(path):
                return path
        raise FileNotFoundError('No anki collection file not found in any of the searched directories.')

    collection_anki2_filename = 'collection.anki2'
    system = platform.system()
    profile_dir = ''

    # get dir - locations may be found at https://docs.ankiweb.net/files.html
    if system == 'Windows':
        modern_dir =        f'{USER_PATH}/AppData/Roaming/Anki2/{profile_name}'
        old_dir =           f'{USER_PATH}/Documents/{profile_name}'
        profile_dir =       get_anki_dir(attempts, modern_dir, old_dir)
    elif system == 'Linux':
        modern_collection = f'{USER_PATH}/.local/share/Anki2/{profile_name}'
        old_collection =    f'{USER_PATH}/Documents/Anki/{profile_name}'
        older_collection =  f'{USER_PATH}/Anki/{profile_name}'
        profile_dir =       get_anki_dir(attempts, modern_collection, old_collection, older_collection)
    elif system == 'Darwin':
        modern_collection = f'{USER_PATH}/Library/Application Support/Anki2/{profile_name}'
        old_collection =    f'{USER_PATH}/Documents/Anki/{profile_name}'
        profile_dir =       get_anki_dir(attempts, modern_collection, old_collection)

    # ensure file exists
    path = f'{profile_dir}/{collection_anki2_filename}'
    if not os.path.exists(path):
        return get_collection_path(profile_name, attempts+1)

    return path


if __name__ == '__main__':
    main()