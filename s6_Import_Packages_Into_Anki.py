import os
import platform
import time

from anki.errors import DBError
from anki.collection import ImportAnkiPackageRequest, Collection

from s5_Generate_Anki_Package import PACKAGE_DIR

#======== Configuraton ===========
PROFILE = 'User 1'                  # default is 'User 1'; see Anki->File->Switch Profile to see what you have. this is case sensitive
PACKAGE_APKG = 'apkg'
PACKAGE_COLAPKG = 'colpkg'
USER_PATH = os.path.expanduser('~')
#=================================


def main():
    import_anki_package()
    return


def import_anki_package():
    #=====
    # get list of packages
    dir_contents = os.listdir(PACKAGE_DIR)
    packages = [f for f in dir_contents if (os.path.isfile(os.path.join(PACKAGE_DIR, f)) and (PACKAGE_APKG in f.lower() or PACKAGE_COLAPKG in f.lower()))]
    print('Found and read packages.')

    # open collection
    collection_anki2_path = get_collection_anki2_path(PROFILE)
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
    return


def get_collection_anki2_path(profile_name):
    collection_anki2_filename = 'collection.anki2'
    system = platform.system()
    profile_dir = ''

    # get dir - locations may be found at https://docs.ankiweb.net/files.html
    if system == 'Windows':
        modern_dir =        f'{USER_PATH}/AppData/Roaming/%APPDATA%/Anki2/{profile_name}'
        old_dir =           f'{USER_PATH}/Documents/{profile_dir}'
        profile_dir = get_anki_dir(modern_dir, old_dir)
    elif system == 'Linux':
        modern_collection = f'{USER_PATH}/.local/share/Anki2/{profile_name}'
        old_collection =    f'{USER_PATH}/Documents/Anki/{profile_name}'
        older_collection =  f'{USER_PATH}/Anki/{profile_name}'
        profile_dir = get_anki_dir(modern_collection, old_collection, older_collection)
    elif system == 'Darwin':
        modern_collection = f'{USER_PATH}/Library/Application Support/Anki2/{profile_name}'
        old_collection =    f'{USER_PATH}/Documents/Anki/{profile_name}'
        profile_dir = get_anki_dir(modern_collection, old_collection)

    # ensure file exists
    path = f'{profile_dir}/{collection_anki2_filename}'
    if not os.path.exists(path):
        raise Exception(FileNotFoundError)

    return path


def get_anki_dir(*anki_dir_tuple):
    for path in anki_dir_tuple:
        if os.path.exists(path):
            return path
    raise Exception(FileNotFoundError)


if __name__ == '__main__':
    main()