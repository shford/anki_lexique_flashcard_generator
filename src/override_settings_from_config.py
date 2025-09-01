import os
from ast import literal_eval


def override_prog_configs_from_file(global_symbol_table) -> None:
    """
    This function exists mostly for the author's edification. You probably don't need a
    config file.

    I kept forgetting to toggle the default values back to what should be the default so
    now we're just going to override them from a local config script.

    I can't git skip tree on just part of a file so... here we are.

    If you really feel like you need a config file the format is:
    {
        'BACKUP': 'True',
        'SomeOtherGlobal': '5',
        'Savvy?': 'Aye',
    }
    :return:
    """
    config_path = '../resources/config.txt'

    # ensure file exists, make file template if it doesn't
    if not os.path.exists(config_path):
        return

    with open(config_path, 'r') as f:
        config_raw = f.read()

        try:
            config_globals_dict = literal_eval(config_raw)

            for key in config_globals_dict.keys():
                if not key == 'comment': # skip comments
                    # only modify if global exists
                    if key not in global_symbol_table:
                        continue

                    # ensure we're importing settings that make sense
                    imported_global_value = literal_eval(config_globals_dict[key])
                    if type(global_symbol_table[key]) != type(imported_global_value):
                        raise ValueError

                    # assign global
                    global_symbol_table[key] = literal_eval(config_globals_dict[key])
        except SyntaxError or ValueError as e:
            print(
                f'\nCredential file at {config_path} is malformed.\nNote: if you delete your file and re-run this program it will remake a sane template.')
            exit(-1)
