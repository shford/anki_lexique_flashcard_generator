"""
@Purpose: Filter lexique lemmes (note: additional correction done dynamically in Step 3).

@Instructions:
  See Github README.

@Note: This is going to be slow-ish. On my machine it took ~1.8 minutes to run.
        But it only needs to run once.
"""
# python core libs
import gc
import os
import time
import warnings
from multiprocessing import Pool, cpu_count

# external library libs
import pandas as pd


# ==== CONFIGURATION ====
# example for alternative directory (works on Nix/Windows), such as to the Desktop
# user_path = os.path.expanduser('~')
# DIR_PATH = f'{user_path}/Desktop' # to the user's Desktop
DIR_PATH = 'resources'
INPUT_CSV = f'{DIR_PATH}/Lexique383.csv'
OUTPUT_DIR = f'default_output'
OUTPUT_CSV = f'{OUTPUT_DIR}/Lexique383 - Filtered.csv'
# ========================

# Globals
desired_POS = ['adj', 'ver', 'adv', 'ono', 'prep', 'con', 'n', 'adj:ind']


def main():
    t1 = time.time()
    clean_lexique()
    t2 = time.time()
    print(f'Cleaned lexique in {t2-t1} seconds.')


def clean_lexique():
    # read in our french dictionary
    word_list = parse_dela_dict()

    # read .csv with pandas
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # rename desired column headers
    df.rename(columns={'1_ortho': 'ortho',
                       '3_lemme': 'lemme',
                       '4_cgram': 'cgram',
                       '5_genre': 'genre',
                       '6_nombre': 'nombre',
                       '7_freqlemfilms2': 'freqlemfilms',
                       '8_freqlemlivres': 'freqlemlivres',
                       '14_islem': 'islem',
                       '28_orthosyll': 'orthosyll',
                       }, inplace=True)

    # remove unwanted columns
    df = df.drop(columns=['2_phon',
                          '9_freqfilms2',
                          '10_freqlivres',
                          '11_infover',
                          '12_nbhomogr',
                          '13_nbhomoph',
                          '15_nblettres',
                          '16_nbphons',
                          '17_cvcv',
                          '18_p_cvcv',
                          '19_voisorth',
                          '20_voisphon',
                          '21_puorth',
                          '22_puphon',
                          '23_syll',
                          '24_nbsyll',
                          '25_cv-cv',
                          '26_orthrenv',
                          '27_phonrenv',
                          '29_cgramortho',
                          '30_deflem',
                          '31_defobs',
                          '32_old20',
                          '33_pld20',
                          '34_morphoder',
                          '35_nbmorph'])

    # filter rows: ortho or lemme column string length > 2 (same as >= 3)
    df = df[df['ortho'].astype(str).str.len() > 2]
    df = df[df['lemme'].astype(str).str.len() > 2]

    # misc removals of known bogus
    df = df[df['lemme'].astype(str) != 'FALSE']
    df = df[df['lemme'].astype(str) != 'TRUE']
    df = df[df['lemme'].astype(str) != 'zzz']
    df = df[df['lemme'].astype(str) != 'zzzz']
    df = df[df['lemme'].astype(str) != 'o']
    df = df[df['lemme'].astype(str) != 'team']
    df = df[df['lemme'].astype(str) != '58e']
    df = df[df['ortho'].astype(str) != 'brunches'] # loan word doesn't take fr plural
    df = df[df['ortho'].astype(str) != 'gardes-chiourme']

    # filter out where ortho is NaN
    df = df[~df['ortho'].isna()]

    # filter out ~6,000 rows where ['ortho'] is not in our dictionary (almost certainly archaic/incorrect/malformed rows)
    df = df[df['ortho'].isin(word_list)]

    # filter out the repeating issue for compound nouns ['hold up', 'hold-up']
    df = df[~df['lemme'].str.contains('-') | (df['lemme'].str.contains('-') & ~df['ortho'].str.contains(' '))]

    # filter out words ending in punctuation - almost certainly artifacts
    df = df[df['ortho'].str[-1] != "'"]

    # rename "cgram" column to .lower()
    df['cgram'] = df['cgram'].str.lower()

    # rename some lexique POS fields to preferred
    df['cgram'] = df['cgram'].replace('nom', 'n')
    df['cgram'] = df['cgram'].replace('pre', 'prep')

    # filter rows for cgram equals desired_POS
    df = df[df['cgram'].isin(desired_POS)]

    # filter for highest priority POS
    df = filter_df_for_highest_pos(df)

    # create a new ODS document
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f'Wrote clean .csv file saved to: {OUTPUT_CSV}')


def parse_dela_dict():
    # return a wordlist from the fr dictionary
    dela_dict_file = 'resources/dela-fr-public.dic'  # UTF-16

    with open(f'{dela_dict_file}', encoding='UTF-16') as f:
        # read everything except comments
        unformatted_valid_words = {line.split('/')[0].strip() for line in f if not line.startswith('#')}

        # remove extraneous information (pos, etc)
        unflattened_valid_word_groups = [x.split('.')[:-1] for x in unformatted_valid_words]

        # flatten list of lists
        flattened_word_groups = flatten_list_of_lists(unflattened_valid_word_groups)

        # split on word groups on commas for individual words
        formatted_words = []
        [formatted_words.extend(split_commas.split(',')) for split_commas in flattened_word_groups]

        # deduplicate words (removes roughly half of all items)
        dedupe_formatted_words = list(set(formatted_words))
        dedupe_formatted_words.remove('')
        dedupe_formatted_words.sort()

        # remove backslashes from compound words
        norm_words = [word.replace('\\-', '-') for word in dedupe_formatted_words]

        # remove uppercase (usually proper nouns) and words with weird characters (probably garbage)
        allowed_words = []
        bad_words = []
        allowed_fr_chars = "abcdefghijklmnopqrstuvwxyzàâäæçéèêëîïôœùûüÿ-’' "
        [(allowed_words.append(word) if (set(word) <= set(allowed_fr_chars)) else bad_words.append(word)) for word in norm_words]
        allowed_words.sort()

    return allowed_words


def flatten_list_of_lists(list_of_lists):
    # fastest way according to stack overflow
    return [x for xs in list_of_lists for x in xs]


def filter_df_for_highest_pos(df) -> pd.DataFrame:
    # add column with POS rank
    df['pos_rank'] = df['cgram'].apply(lambda pos: desired_POS.index(pos) if pos in desired_POS else len(desired_POS))

    # get lemmes and their corresponding rows
    df_cols = df.columns
    lemmes, lemme_to_df_lookup = group_dfs_by_lemme(df, df_cols)

    # free old df - we'll be reconstructing it from scratch
    del df
    gc.collect()

    # populate df with only highest priority POS for each lemme
    # note: use list for intermediate step to avoid pd.concat() O(n^2) slow down inside loop
    tmp_df_list = []
    for l in lemmes:
        # get all rows associated with this lemme
        lemme_df = lemme_to_df_lookup[l]

        # filter
        min_rank = lemme_df['pos_rank'].min()
        lemme_df = lemme_df[lemme_df['pos_rank'] == min_rank]

        # append filtered rows to df
        lemme_list = lemme_df.to_dict(orient='records')
        tmp_df_list.extend(lemme_list)

    # convert intermediate into a proper dataframe
    df = pd.DataFrame(tmp_df_list, columns=df_cols)

    # remove unneeded POS rank
    df = df.drop(columns=['pos_rank'])

    return df


def group_dfs_by_lemme(df, df_cols) -> (list, dict):
    """
    Create a list of unique lexique lemmes.
    Create a dict of rows belonging to each lemme.
    Example:
        [ "lemme1", "lemme2", "lemme3" ]
        {"lemme1":[row_A, row_B], "lemme2":[row_C,...]}
    """
    unique_lemmes = []
    lemme_to_df_lookup = {}
    lemme_to_list_lookup = {}

    # iterrows, itertuples, to_dict, and zip all suck roughly equally
    # df_list = df.to_dict(orient='records')
    # for row in df_list:
    #     lemme = row['lemme']
    #     if lemme not in lemme_to_list_lookup:
    #         unique_lemmes.append(lemme)
    #         lemme_to_list_lookup[lemme] = []
    #     lemme_to_list_lookup[lemme].append(list(row.values()))

    for c1,c2,c3,c4,c5,c6,c7,c8,c9,c10 in zip(df['ortho'], df['lemme'], df['cgram'], df['genre'],
                                       df['nombre'],df['freqlemfilms'], df['freqlemlivres'],
                                       df['islem'], df['orthosyll'], df['pos_rank']):
        if c2 not in lemme_to_list_lookup:
            unique_lemmes.append(c2)
            lemme_to_list_lookup[c2] = []
        lemme_to_list_lookup[c2].append([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10])

    for lemme in unique_lemmes:
        lemme_to_df_lookup[lemme] = pd.DataFrame(lemme_to_list_lookup[lemme], columns=df_cols) # 2D axis will convert nicely

    return unique_lemmes, lemme_to_df_lookup


if __name__ == '__main__':
    # ignore pandas 'FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        main()
