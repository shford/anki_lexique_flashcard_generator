"""

Note 1: The formatting for adjectives and nouns is surprisingly inconvenient to decouple.
        It would require a lot of duplicate code so in format_noun_declension_adj() we actually
        may format_noun_declension_nom(). The adjective does not magically become a noun. I just
        didn't want to abstract out the rows 2 & 3 special cases (thanks French) and corrective logic.
        Likewise if you see a ['cgram'] == 'adj' inside the _nom() then now you know it's just there
        for this. Sorry eh.

Note 2: The Lexique 3.83 excel was already sorted such that nombre 's' came prior to 'p'.
        I doubt that'll change but if this breaks for future versions then it be worth adding
        a quick sort (not actually quicksort, jeez) to do that within the POS for each lemme.

"""
import os
from ast import literal_eval
import deepl
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# TODO
#   - get rid or 'adj'/'nom' checks EXCEPT for homophone/homonym/x2 lemma's (should fix 7001-7500)
#   - there's probably more irregular formats, but 1-10,000 look good
#   - add column for the portion that we want to send to the API
#   - add Tags for irregular cards that won't meet formatting


# === Configuration variables ===
USER_PATH = os.path.expanduser('~')
INPUT_CSV = f'{USER_PATH}/Documents/flashcard_project_new/lexique_exported_files/Freq 10501 - 11500.csv'
OUTPUT_DIR = f'{USER_PATH}/Documents/flashcard_project_new/anki_lexique_imports'
OUTPUT_PREFIX = 'anki_deck_'
CHUNK_SIZE = 500

# === End Config ===

# POS priority for sorting and filtering
POS_PRIORITY = ['adj', 'adv', 'pre', 'ver', 'ono', 'nom', 'con']


@dataclass
class Row_Stats:
    def __init__(self, rows):
        self.rows_m = rows[rows['genre'] == 'm']
        self.rows_f = rows[rows['genre'] == 'f']
        self.rows_s = rows[rows['nombre'] == 's']
        self.rows_p = rows[rows['nombre'] == 'p']
        self.rows_genre_na = rows[rows['genre'].isna()]
        self.rows_nombre_na = rows[rows['nombre'].isna()]
        self.rows_f_s = rows[(rows['genre'] == 'f') & (rows['nombre'] == 's')]
        self.rows_m_s = rows[(rows['genre'] == 'm') & (rows['nombre'] == 's')]
        self.rows_f_p = rows[(rows['genre'] == 'f') & (rows['nombre'] == 'p')]
        self.rows_m_p = rows[(rows['genre'] == 'm') & (rows['nombre'] == 'p')]
        self.rows_f_na = rows[(rows['genre'] == 'f') & (rows['nombre'].isna())]
        self.rows_m_na = rows[(rows['genre'] == 'm') & (rows['nombre'].isna())]
        self.rows_na_s = rows[(rows['genre'].isna()) & (rows['nombre'] == 's')]
        self.rows_na_p = rows[(rows['genre'].isna()) & (rows['nombre'] == 'p')]
        self.rows_na_na = rows[(rows['genre'].isna()) & (rows['nombre'].isna())]

        self.num_m = len(self.rows_m)
        self.num_f = len(self.rows_f)
        self.num_s = len(self.rows_s)
        self.num_p = len(self.rows_p)
        self.num_genre_na = len(self.rows_genre_na)
        self.num_nombre_na = len(self.rows_nombre_na)
        self.num_f_s = len(self.rows_f_s)
        self.num_m_s = len(self.rows_m_s)
        self.num_f_p = len(self.rows_f_p)
        self.num_m_p = len(self.rows_m_p)
        self.num_f_na = len(self.rows_f_na)
        self.num_m_na = len(self.rows_m_na)
        self.num_na_s = len(self.rows_na_s)
        self.num_na_p = len(self.rows_na_p)
        self.num_na_na = len(self.rows_na_na)


def main():
    formatting_exception_count = 0

    # prep for DeepL translations
    auth_key = read_creds()
    deepl_client = deepl.DeepLClient(auth_key)
    source_language = 'FR'
    target_language = 'EN-US'

    # load pandas
    df = pd.read_csv(INPUT_CSV, encoding='utf-8')

    # calculate starting frequency index from filename
    freq_start = parse_start_frequency(INPUT_CSV)

    # ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # create a set of {"lemme1":[row_A, row_B], "lemme2":[row_X,...]} & ["lemme1", "lemme2", ...]
    lemmes, lemme_to_rows = group_rows_by_lemme(df)

    # process lemme in chunks of CHUNK_SIZE
    for chunk_idx in range(0, len(lemmes), CHUNK_SIZE):
        export_rows = []
        lemme_chunk = lemmes[chunk_idx : chunk_idx + CHUNK_SIZE]

        for lemme in lemme_chunk:
            lemme_df = pd.DataFrame(lemme_to_rows[lemme])
            pos = lemme_df['cgram'].iloc[0]

            # get DeepL English translation
            # translation = translate(lemme, pos, deepl_client, source_language, target_language)
            translation = ''

            # format 'Noun Declension' field
            noun_decls = format_noun_declension(lemme, lemme_df, pos)

            # if formatting fails, print all rows for this lemme and POS with nombre and ortho for debug
            if noun_decls is None:
                formatting_exception_count += 1
                print_formatting_exceptions(lemme, pos, lemme_df)

                # just treat lemme as singular adj - e.g. give up and settle for bolding the lemme
                noun_decls = singular_bold(lemme, 'adj')

            # copy orthosyll column to pronunciation column - use matching lemme orthosyll if available
            pronunciation = get_pronunciation(lemme, lemme_df)

            # update rows
            update_to_export_rows(lemme, pos, noun_decls, pronunciation, translation, export_rows)

        # write sheet to be imported into Anki
        write_anki_csv(freq_start, chunk_idx, lemme_chunk, export_rows, formatting_exception_count)


def read_creds():
    """
    :return: DeepL API key
    """
    abs_path = os.path.abspath(os.getcwd())
    credentials_file = 'resources/project_credentials.txt'

    # ensure file exists, make file template if it doesn't
    if not os.path.exists(credentials_file):
        with open(credentials_file, 'w+') as f:
            cred_template = ("{\n"
                             "\t'deepl_api_key': 'copy_past_key_here_from_website__DO_NOT_REMOVE_SURROUNDING_QUOTES'\n"
                             "}"
                             )
            f.write(cred_template)
        print(f'\nPlease fill in credentials file template created at: {abs_path}/{credentials_file}')
        exit(-1)

    # read credentials
    with open(credentials_file, 'r') as f:
        creds_raw = f.read()
        cred_dict = None
        try:
            cred_dict = literal_eval(creds_raw)
            deepl_api_key = cred_dict['deepl_api_key']
            return deepl_api_key
        except SyntaxError or ValueError as e:
            print(f'\nCredential file at {abs_path}/{credentials_file} is malformed. Please check your credentials file.\nNote: if you delete your file and re-run this program it will remake the template. After, copy/paste your credentials where indicated.')
            exit(-1)


def translate(lemme, pos, deepl_client, source_lang, target_language) -> str:
    """
    :return: DeepL translation
    """
    deepl_translation = deepl_client.translate_text(lemme, source_lang=source_lang, target_lang=target_language)
    if pos == 'ver':
        # prepend 'to '
        deepl_translation = f'to {deepl_translation}'

    return deepl_translation.lower()


def group_rows_by_lemme(df) -> (list, dict):
    """
    Create a list of unique lexique lemmes.
    Create a dict of rows belonging to each lemme.
    Example:
        [ "lemme1", "lemme2", "lemme3" ]
        {"lemme1":[row_A, row_B], "lemme2":[row_C,...]}
    """
    unique_lemmes = []
    lemme_row_lookup = {}

    for idx, row in df.iterrows():
        lemme = row['lemme']
        if lemme not in lemme_row_lookup:
            lemme_row_lookup[lemme] = []
            unique_lemmes.append(lemme)
        lemme_row_lookup[lemme].append(row)

    return unique_lemmes, lemme_row_lookup


def parse_start_frequency(filename) -> int:
    """
    :return: frequency start index from filename like 'Freq 1-500.csv'
    """
    match = re.search(r'Freq (\d+) - \d+', filename)
    if not match:
        raise ValueError('Invalid filename: ' + filename + '\n\nFilename is required to match format to accuracately determine frequency index.')
    return int(match.group(1))


def format_noun_declension(lemme, rows, pos):
    """
    :return: correct formatting rule.
    """
    # we'll just pre-compute this junk. it's a little inefficient but it reduces code complexity
    row_stats = Row_Stats(rows=rows)

    # check for hard-coded exceptions first
    hard_coded_format = handle_hard_coded_formats(rows, lemme)
    if hard_coded_format is not None or hard_coded_format is False:
        return hard_coded_format

    if pos in {'ver', 'adv', 'pre', 'con', 'ono'}:
        return singular_bold(lemme, pos)
    else:
        num_rows = len(rows)
        try:
            # if 1 row OR genre empty and all ortho's are equal then treat as single
            if num_rows == 1 or (all(x == lemme for x in rows['ortho']) and rows['genre'].isna().all()):
                return row1_func(rows, row_stats, lemme, pos)
            elif num_rows == 2:
                return row2_func(rows, row_stats, lemme, pos)
            elif num_rows == 3:
                return row3_func(rows, row_stats, lemme, pos)
            elif num_rows == 4:
                return row4_func(rows, row_stats, lemme, pos)
            elif num_rows == 5:
                return row5_func(rows, row_stats, lemme, pos)
            else:
                return None
        except ValueError or TypeError:
            return None


def row1_func(rows, rs, lemme, pos):
    # single row cases
    row = rows.iloc[0]
    genre =  row.get('genre', np.nan)
    nombre = row.get('nombre', np.nan)
    ortho = row['ortho']

    if rs.num_p == 1:
        # assume this means only a plural form exists (e.g. you can have 'pants' but not 'pant')
        #   genre   nombre
        #     *       p
        return plural_bold(ortho, pos, genre)
    else:
        if rs.num_genre_na == 1:
            # treat as adet, unknown article
            #   genre   nombre
            #     _      (s|_)
            return singular_bold(ortho, 'adj')
        elif rs.num_s == 1:
            # assume only a singular form exists
            #   genre   nombre
            #   (m|f)     s
            return singular_bold(ortho, pos, genre)
        else:
            # assume this is supposed to be both a single & plural form... this could be wrong but based on what I've seen that's most common correct fix for this Lexique error
            #   genre   nombre
            #   (m|f)     _
            return sp_bold(lemme, pos, ortho, ortho, genre)


def row2_func(rows, rs, lemme, pos):
    """
    :return: formatting per rules
    """
    # infer missing 'nombre'
    if rs.num_nombre_na == 1:
        idx = rows[rows['nombre'].isna()].index[0]
        nombres = rows['nombre'].dropna()
        # if one is singular, then the other must be plural
        if rs.num_s == 1:
            rows.at[idx, 'nombre'] = 'p'
        # if one is plural, then the other must be singular
        elif rs.num_p == 1:
            rows.at[idx, 'nombre'] = 's'
        else:
            return None

    # infer missing 'genre'
    if rs.num_genre_na == 1:
        idx = rows[rows['genre'].isna()].index[0]
        nombres = rows['genre'].dropna()
        # if one is male, then the other must be male
        if rs.num_m == 1:
            rows.at[idx, 'genre'] = 'm'
        # if one is fem, then the other must be fem
        elif rs.num_f == 1:
            rows.at[idx, 'genre'] = 'f'
        else:
            return None

    # refresh row stats after fixes
    rs = Row_Stats(rows)
    row1 = rows.iloc[0]
    row2 = rows.iloc[1]
    r1_genre = row1['genre']
    r1_nombre = row1['nombre']
    r2_genre = row2['genre']
    r2_nombre = row2['nombre']

    # 'adj' or 'nom'
    if rs.num_m_s == 1 and rs.num_f_s == 1:
        #   genre   nombre  |   genre   nombre
        #     m       s     |     f       s
        #     f       s     |     s       s
        return ms_fs_bold(lemme, pos, rs.rows_m_s.iloc[0]['ortho'], rs.rows_f_s.iloc[0]['ortho'])

    # nom
    if pos == 'nom':
        # homophone, homonym, diff lemmas
        if rs.num_genre_na == 2 and rs.num_s == 1 and rs.num_p == 1:
            """
            This logic handles where for both rows genre is NaN but one row has 's' and one row has 'p'.
            The Lexique, in its grand wisdom, has decided this symobolizes words that are homographs and homophones but have different lemmas.

            They are semantically distinct words that just happen to share the same spelling and pronunciation but have different meanings depending on gender. 
            These will require their own flashcards and thus need multiple entries.
            Now we get to return lists - yay.

            For example:
                la tour (tower) v. le tour (turn)
                la livre (pound) v. le livre (book)
            """
            ortho_s = rs.rows_s.iloc[0]['ortho']
            ortho_p = rs.rows_p.iloc[0]['ortho']
            return [sp_bold(lemme, pos, ortho_s, ortho_p, 'm'),
                    sp_bold(lemme, pos, ortho_s, ortho_p, 'f')]

        # nouns with conflicting genres across different plurality are rare, but valid and modern..
        if rs.num_m_s == 1 and rs.num_f_p == 1:
            #   genre   nombre
            #     m       s
            #     f       p
            ortho_s = rs.rows_s.iloc[0]['ortho']
            ortho_p = rs.rows_p.iloc[0]['ortho']
            return sp_bold(lemme, pos, ortho_s, ortho_p, 'ms_fp')
        elif rs.num_f_s == 1 and rs.num_m_p == 1:
            #   genre   nombre
            #     f       s
            #     m       p
            ortho_s = rs.rows_s.iloc[0]['ortho']
            ortho_p = rs.rows_p.iloc[0]['ortho']
            return sp_bold(lemme, pos, ortho_s, ortho_p, 'fs_mp')

        # regular formatting for male
        if rs.num_f == 0 and rs.num_m >= 1:
            if rs.num_p == 1:
                ortho_s = rs.rows_s.iloc[0]['ortho'] if rs.num_s == 1 else rs.rows_nombre_na.iloc[0]['ortho']
                ortho_p = rs.rows_p.iloc[0]['ortho']
                return sp_bold(lemme, pos, ortho_s, ortho_p, 'm')
            elif rs.num_s == 1:
                ortho_s = rs.rows_s.iloc[0]['ortho']
                ortho_p = rs.rows_p.iloc[0]['ortho'] if rs.num_p == 1 else rs.rows_nombre_na.iloc[0]['ortho']
                return sp_bold(lemme, pos, ortho_s, ortho_p, 'f')
            else:
                return sp_bold(lemme, pos, row1['ortho'], row2['ortho'], 'm')

        # regular formatting for fem.
        if rs.num_m == 0 and rs.num_f >= 1:
            if rs.num_p == 1:
                ortho_s = rs.rows_s.iloc[0]['ortho'] if rs.num_s == 1 else rs.rows_nombre_na.iloc[0]['ortho']
                ortho_p = rs.rows_p.iloc[0]['ortho']
                return sp_bold(lemme, pos, ortho_s, ortho_p, 'f')
            elif rs.num_s == 1:
                ortho_s = rs.rows_s.iloc[0]['ortho']
                ortho_p = rs.rows_p.iloc[0]['ortho'] if rs.num_p == 1 else rs.rows_nombre_na.iloc[0]['ortho']
                return sp_bold(lemme, pos, ortho_s, ortho_p, 'f')
            else:
                return sp_bold(lemme, pos, row1['ortho'], row2['ortho'], 'f')

    # adj
    if pos == 'adj':
        if rs.num_genre_na == 2 and rs.num_s == 1 and rs.num_p == 1:
            #   genre   nombre
            #     _       s
            #     _       p
            ortho_s = rs.rows_s.iloc[0]['ortho']
            ortho_p = rs.rows_p.iloc[0]['ortho']
            return sp_bold(lemme, pos, ortho_s, ortho_p)

        if rs.num_genre_na == 2 and rs.num_nombre_na == 2:
            #   genre   nombre
            #     _       _
            #     _       _
            return sp_bold(lemme, pos, row1['ortho'], row2['ortho'])  # gamble that the corpus row order is correct

        # regular formatting
        if rs.num_p == 1:
            ortho_s = rs.rows_s.iloc[0]['ortho'] if rs.num_s == 1 else rs.rows_nombre_na.iloc[0]['ortho']
            ortho_p = rs.rows_p.iloc[0]['ortho']
            return sp_bold(lemme, pos, ortho_s, ortho_p)
        else:
            return sp_bold(lemme, pos, row1['ortho'], row2['ortho'])

    return None


def row3_func(rows, rs, lemme, pos):
    # correct missing if able
    if rs.num_nombre_na <= 1 and rs.num_genre_na <= 1:
        # infer missing 'nombre'
        if rows['nombre'].isna().any():
            idx = rows[rows['nombre'].isna()].index[0]
            nombres = rows['nombre'].dropna()
            # if we have 2 singulars, then the other must be plural
            if rs.num_s == 2:
                rows.at[idx, 'nombre'] = 'p'
            # if we have 1 singular and 1 plural, then the other must be singular
            elif rs.num_s == 1 and rs.num_p == 1:
                rows.at[idx, 'nombre'] = 's'

        # infer missing 'genre' (only if 'nombre' is 's')
        if rows['genre'].isna().any():
            idx = rows[rows['genre'].isna()].index[0]
            if rows.at[idx, 'nombre'] == 'p':
                pass  # genre doesn't matter for plural
            else:
                other = rows[(rows.index != idx) & (rows['nombre'] == 's')]
                genres = other['genre'].dropna().unique()
                if len(genres) == 1:
                    rows.at[idx, 'genre'] = 'f' if genres[0] == 'm' else 'm'

    # update row stats
    rs = Row_Stats(rows)
    row1 = rows.iloc[0]
    row2 = rows.iloc[1]
    row3 = rows.iloc[2]

    if pos == 'nom' or pos == 'adj':
        if rs.num_m_s == 1 and rs.num_m_p == 1 and rs.num_f_p == 1:
            # this is going to be an exception where we're not going to choose one
            # in this case we got ['tueur', 'tueurs', 'tueuses'] which ought to be
            # ['tueur', 'tueuse', 'tueurs', 'tueuses']
            # so we're going to try to infer the fs. 's' and pass these to four_det_bold()
            ortho_m_s = rs.rows_m_s.iloc[0]['ortho']
            ortho_f_s = rs.rows_f_p.iloc[0]['ortho'][:-1]
            ortho_m_p = rs.rows_m_p.iloc[0]['ortho']
            ortho_f_p = rs.rows_f_p.iloc[0]['ortho']
            return four_bold(lemme, pos, ortho_m_s, ortho_f_s, ortho_m_p, ortho_f_p)
    elif pos == 'nom':
        #     genre       nombre
        # ( m | f | _ )     p
        # ( m | f | _ )     p
        # ( m | f | _ )     p
        if rs.num_p == 3 and (rs.num_m == 3 or rs.num_f == 3 or rs.num_genre_na == 3):
            # if all three are the same and plural
            if row1['ortho'] == row2['ortho'] == row3['ortho']:
                # assume only the plural exists
                return plural_bold(row1['ortho'], pos, row1['genre'])
            else:
                # else assume it's malformed gibberish
                return None
        elif rs.num_s == 2 and rs.num_p == 1:
            if rs.num_f_s == 1 and rs.num_na_s == 1 and rs.num_na_p == 1:
                #   genre   nombre
                #     _       s
                #     f       s
                #     _       p
                ortho_m = rs.rows_na_s.iloc[0]['ortho']
                ortho_p = rs.rows_na_p.iloc[0]['ortho']
                ortho_f = rs.rows_f_s.iloc[0]['ortho']
                return mpf_det_bold(lemme, pos, ortho_m, ortho_p, ortho_f)
            elif rs.num_m_s == 1 and rs.num_na_s == 1 and rs.num_na_p == 1:
                #   genre   nombre
                #     m       s
                #     _       s
                #     _       p
                ortho_m = rs.rows_m_s.iloc[0]['ortho']
                ortho_p = rs.rows_na_p.iloc[0]['ortho']
                ortho_f = rs.rows_na_s.iloc[0]['ortho']
                return mpf_det_bold(lemme, pos, ortho_m, ortho_p, ortho_f)
        elif rs.num_s == 1 and rs.num_p == 2:
            ortho_s = rs.rows_s.iloc[0]['ortho']
            if rs.num_m == 3 or rs.num_f == 3:
                # resolve down to one (hopefully more grammatically correct) plural
                #   genre   nombre  |   genre   nombre
                #     f       s     |     m       s
                #     f       p     |     m       p
                #     f       p     |     m       p
                if rs.num_m == 3:
                    genre = 'm'
                    p_row1 = rs.rows_m_p.iloc[0]['ortho']
                    p_row2 = rs.rows_m_p.iloc[1]['ortho']
                else:
                    genre = 'f'
                    p_row1 = rs.rows_f_p.iloc[0]['ortho']
                    p_row2 = rs.rows_f_p.iloc[1]['ortho']

                if '-' in lemme:
                    # for compound-nouns, the plurality adj. must agree w/ the noun ergo the longer word (grand-mere [archaic/wrong] v. grands-mere [correct]) will be the correct one due to the addition of the 's' prior to the '-'
                    if len(p_row1) > len(p_row2):
                        return sp_bold(lemme, pos, ortho_s, p_row1, genre)
                    elif len(p_row2) > len(p_row1):
                        return sp_bold(lemme, pos, ortho_s, p_row2, genre)
                    else:
                        return None # could not determine correct plural
                else:
                    # if it's not a compound noun let's just choose the plural that ends in 's'
                    # ['scénarii', 'scénario', 'scénarios']
                    if p_row1[-1:] == 's' and p_row2[-1:] != 's':
                        return sp_bold(lemme, pos, ortho_s, p_row1, genre)
                    if p_row2[-1:] == 's' and p_row1[-1:] != 's':
                        return sp_bold(lemme, pos, ortho_s, p_row2, genre)
            else:
                pass
    elif pos == 'adj':
        pass # no adj specific corrections to make for three rows

    # identify forms
    masc_sing = rows[(rows['genre'] == 'm') & (rows['nombre'] == 's')]
    fem_sing = rows[(rows['genre'] == 'f') & (rows['nombre'] == 's')]
    plural = rows[rows['nombre'] == 'p']

    if len(masc_sing) != 1 or len(plural) != 1:
        return None
    if len(fem_sing) > 1:
        return None

    ortho_m = masc_sing.iloc[0]['ortho']
    ortho_p = plural.iloc[0]['ortho']
    ortho_f = fem_sing.iloc[0]['ortho'] if not fem_sing.empty else None

    # if both plural and feminine are the same as masculine, return None
    if (ortho_p == ortho_m) and (ortho_f is None or ortho_f == ortho_m):
        return None

    # only format if both forms differ
    if ortho_p != ortho_m and ortho_f and ortho_f != ortho_m:
        return mpf_det_bold(lemme, pos, ortho_m, ortho_p, ortho_f)
    elif ortho_p != ortho_m and (ortho_f is None or ortho_f == ortho_m):
        return sp_bold(lemme, pos, ortho_m, ortho_p, 'm')
    elif ortho_f and ortho_f != ortho_m and ortho_p == ortho_m:
        return sp_bold(lemme, pos, ortho_f, ortho_p,'f')
    else:
        return None


def row4_func(rows, rs, lemme, pos):
    if pos == 'nom':
        # count missing fields per row
        missing_genre_mask = rows['genre'].isna()
        missing_nombre_mask = rows['nombre'].isna()
        missing_both_mask = missing_genre_mask & missing_nombre_mask

        # total missing counts
        total_missing_genre = missing_genre_mask.sum()
        total_missing_nombre = missing_nombre_mask.sum()
        total_missing_both = missing_both_mask.sum()

        # sanity checks
        if total_missing_both > 1:
            return None  # more than one row missing both - ambiguous

        # if there is one row missing both genre and nombre
        if total_missing_both == 1:
            # check that the other three rows have no missing fields
            others = rows[~missing_both_mask]
            if others['genre'].isna().any() or others['nombre'].isna().any():
                return None  # Others must be complete

            # check others cover 3 distinct (genre,nombre) combos
            combos = set(zip(others['genre'], others['nombre']))
            if len(combos) != 3:
                return None  # not unique combos, can't infer

            # determine the missing combo (genre,nombre)
            expected_combos = {('m', 's'), ('m', 'p'), ('f', 's'), ('f', 'p')}
            missing_combo = expected_combos - combos
            if len(missing_combo) != 1:
                return None  # ambiguous missing combo

            missing_genre, missing_nombre = missing_combo.pop()
            # assign missing fields to the missing_both row
            idx = rows[missing_both_mask].index[0]
            rows.at[idx, 'genre'] = missing_genre
            rows.at[idx, 'nombre'] = missing_nombre

        else:
            # no rows missing both fields
            # handle missing single fields (genre or nombre)

            # infer missing nombre if exactly one missing
            if total_missing_nombre == 1:
                idx = rows[missing_nombre_mask].index[0]
                existing_nombres = set(rows.loc[~missing_nombre_mask, 'nombre'])
                expected_nombres = {'s', 'p'}
                missing_nombre_values = expected_nombres - existing_nombres
                if len(missing_nombre_values) != 1:
                    return None
                rows.at[idx, 'nombre'] = missing_nombre_values.pop()

            # infer missing genre if exactly one missing
            if total_missing_genre == 1:
                idx = rows[missing_genre_mask].index[0]
                row_nombre = rows.at[idx, 'nombre']
                if row_nombre not in {'s', 'p'}:
                    return None  # nombre must be known to infer genre
                same_nombre_rows = rows[(rows.index != idx) & (rows['nombre'] == row_nombre)]
                existing_genres = set(same_nombre_rows['genre'].dropna())
                expected_genres = {'m', 'f'}
                missing_genres = expected_genres - existing_genres
                if len(missing_genres) != 1:
                    return None
                rows.at[idx, 'genre'] = missing_genres.pop()

        # after inference, if any genre or nombre is still missing, return None
        if rows['genre'].isna().any() or rows['nombre'].isna().any():
            return None

        # validate that all (genre, nombre) combinations are unique and complete
        combos_seen = set()
        groups = {
            ('m', 's'): None,
            ('m', 'p'): None,
            ('f', 's'): None,
            ('f', 'p'): None,
        }

        for _, row in rows.iterrows():
            key = (row['genre'], row['nombre'])
            if key not in groups:
                return None  # invalid genre/number combo
            if groups[key] is not None:
                # this means we have a duplicate key - try fix
                if rs.num_m_s == 1 and rs.num_f_s == 2 and rs.num_m_p == 1 and rs.num_f_p == 0:
                    # most likely this should be three row and
                    # there's an archaic and a modern fem. sing, attempt to select modern form
                    row_fs1 = rs.rows_f_s.iloc[0]['ortho']
                    row_fs2 = rs.rows_f_s.iloc[1]['ortho']
                    ortho_m = rs.rows_m_s.iloc[0]['ortho']
                    ortho_p = rs.rows_m_p.iloc[0]['ortho']
                    modern_suffix = 'euse'
                    if row_fs1[-len(modern_suffix):] == modern_suffix:
                        return mpf_det_bold(lemme, pos, ortho_m, ortho_p, row_fs1)
                    elif row_fs2[-len(modern_suffix):] == modern_suffix:
                        return mpf_det_bold(lemme, pos, ortho_m, ortho_p, row_fs2)
                    else:
                        return None

                return None  # unfixable duplicate entry
            groups[key] = row['ortho']

        # if any combo missing, return None
        if any(v is None for v in groups.values()):
            return None

        return four_bold(lemme, pos, groups[('m', 's')], groups[('m', 'p')], groups[('f', 's')], groups[('f', 'p')])

    if pos == 'adj':
        genre_vals = rows['genre'].dropna().unique()
        nombre_vals = rows['nombre'].dropna().unique()

        if len(rows) == 4:
            # Expect ms, mpl, fs, fpl
            ms = find_row(rows, 'm', 's')
            mpl = find_row(rows, 'm', 'p')
            fs = find_row(rows, 'f', 's')
            fpl = find_row(rows, 'f', 'p')

            if rs.num_na_s == 1 and rs.num_na_p == 1 and rs.num_f_s == 1 and rs.num_f_p == 1:
                # works for 'adj' and 'nom' - assumes male is more likely than two archaics
                #   genre   nombre
                #     _       s     ->  m   s
                #     _       p     ->  m   p
                #     f       s
                #     f       p
                ortho_ms = rs.rows_na_s.iloc[0]['ortho']
                ortho_mp = rs.rows_na_p.iloc[0]['ortho']
                return four_bold(lemme, pos, ortho_ms, ortho_mp,
                                 rs.rows_f_s.iloc[0]['ortho'], rs.rows_f_p.iloc[0]['ortho'])

            if ms is not None and mpl is not None and fs is not None and fpl is not None:
                # nothing missing
                return four_bold(lemme, pos, ms['ortho'], mpl['ortho'], fs['ortho'], mpl['ortho'])
            else:
                # one row missing
                if (ms is not None and mpl is not None and fs is not None) or (ms is not None and mpl is not None and fpl is not None) or (ms is not None and fs is not None and fpl is not None) or (mpl is not None and fs is not None and fpl is not None):
                    # assign ms|mpl|fs|fpl to row with missing genre/nombre malformed - process of elimination
                    malformed_row = rows[rows['genre'].isna() | rows['nombre'].isna()]
                    if ms is not None and mpl is not None and fs is not None:
                        fpl = malformed_row
                    elif ms is not None and mpl is not None and fpl is not None:
                        fs = malformed_row
                    elif ms is not None and fs is not None and fpl is not None:
                        mpl = malformed_row
                    else:
                        ms = malformed_row

                    # return corrected value
                    return four_bold(lemme, pos, ms['ortho'], mpl['ortho'], fs['ortho'], mpl['ortho'])
                # todo could fix infer more fixes, for example if ms and fpl were both missing but there were rows with m_ and _pl

    return None


def row5_func(rows, rs, lemme, pos):
    if rs.num_m_s == 1 and rs.num_f_s == 1 and rs.num_m_p == 2 and rs.num_f_p == 1:
        # most likely there's an archaic and a modern masc. plural, attempt to select modern form
        row_mp1 = rs.rows_m_p.iloc[0]['ortho']
        row_mp2 = rs.rows_m_p.iloc[1]['ortho']
        archaic_ending = 'als'
        if archaic_ending == row_mp1[-3:]:
            return four_bold(lemme, pos, rs.rows_m_s.iloc[0], rs.rows_f_s.iloc[0], row_mp2, rs.rows_f_p.iloc[0])
        elif archaic_ending in row_mp2[-3:]:
            return four_bold(lemme, pos, rs.rows_m_s.iloc[0], rs.rows_f_s.iloc[0], row_mp1, rs.rows_f_p.iloc[0])
        else:
            return None
    elif rs.num_m_s == 1 and rs.num_f_s == 2 and rs.num_m_p == 1 and rs.num_f_p == 1:
        # most likely there's an archaic and a modern fem. singular, attempt to select modern form
        row_fs1 = rs.rows_f_s.iloc[0]['ortho']
        row_fs2 = rs.rows_f_s.iloc[1]['ortho']
        archaic_ending = 'eresse'
        if archaic_ending == row_fs1[-6:]:
            return four_bold(lemme, pos, rs.rows_m_s.iloc[0], row_fs2, rs.rows_m_p.iloc[0], rs.rows_f_p.iloc[0])
        elif archaic_ending == row_fs2[-6:]:
            return four_bold(lemme, pos, rs.rows_m_s.iloc[0], row_fs1, rs.rows_m_p.iloc[0], rs.rows_f_p.iloc[0])
        else:
            return None

    return None


def handle_hard_coded_formats(rows, lemme):
    """
    :return: pretty self explanatory
    note: returning false prevents duplicates from getting exported.
            DO NOT CHANGE False TO None.
    """
    # exceptions because of duplicate lemme entries in lexique that should be consolidated into one card
    HARD_CODED_ADJ_4_ROWS = {
        'tout', 'toute', 'tous', 'toutes',
        'aucun', 'aucune', 'aucuns', 'aucunes',
        'quelque', 'quelques',
    }
    if lemme in HARD_CODED_ADJ_4_ROWS:
        if lemme == 'tout':
            return four_bold(lemme, 'adj', 'tout', 'tous', 'toute', 'toutes')
        elif lemme == 'aucun':
            return four_bold(lemme, 'adj', 'aucun', 'aucuns', 'aucune', 'aucunes')
        elif lemme == 'quelque':
            return sp_bold('quelque', 'adj', 'quelque', 'quelques')
        else:
            return False

    # exception because singular ortho is missing from lexique
    if lemme == 'fois':
        return sp_bold(lemme, 'nom', lemme, lemme, 'f')

    # exceptions for unique/archaic/rare poetic spellings (that can't be corrected easily w/ rules)
    if lemme == 'oeil':
        return sp_bold(lemme, 'nom', lemme, 'yeux', 'm')
    elif lemme == 'lieu':
        return sp_bold(lemme, 'nom', lemme, 'lieux', 'm')
    elif lemme == 'aïeul':
        return '<b>aïeul</b> [<gr><i>ms. </i></gr> <blue>le aïeul</blue>; <gr><i>mpl. (refers to male members of a genealogical tree - literal grandfathers/forefathers)</i></gr> <blue>les aïeuls</blue>; <gr><i>mpl. (refers to collective ancestors regardless of gender even if not from a single literal bloodline)</i></gr> <blue>les aïeux</blue>; <gr><i>fs. </i></gr> <red>la aïeule</red>; <gr><i>fpl. </i></gr> archaic]'

    return None


def singular_bold(ortho_s, pos, genre=None) -> str:
    """
    :return: formatted string for one singular
    """
    if pos == 'nom':
        if genre is None:
            raise Exception('Invalid arguments passed to singular_bold() for where pos is nom')
        elif genre == 'm':
            c_male = apply_contraction(f'le {ortho_s}')
            return f'<b><blue>{c_male}</blue></b>'
        elif genre == 'f':
            c_fem = apply_contraction(f'la {ortho_s}')
            return f'<b><red>{c_fem}</red></b>'
        else:
            raise Exception('Invalid genre passed to to det function: singular_bold()')
    else:
        return f'<b>{ortho_s}</b>'


def plural_bold(plural, pos, genre=None) -> str:
    """
    :return: formatted string for one plural
    """
    if pos == 'nom':
        if genre == 'm':
            return f"<b><blue>les {plural}</blue></b>"
        elif genre == 'f':
            return f"<b><red>les {plural}</red></b>"
        else:
            return f"<b>les {plural}</b>"
    else:
        return f'<b>{plural}</b>'


def ms_fs_bold(lemme, pos, ortho_ms, ortho_fs) -> str:
    """
    :return: formatted one masculine singular and one feminine singular
    """
    # shout [grand-papa, grand-mama]
    if pos == 'nom':
        c_male = apply_contraction(f'le {ortho_ms}')
        return f'<gr><i>ms. </i></gr> <blue><b>{c_male}</b></blue>; <gr><i>fs. </i></gr> <red><b>la {ortho_fs}</red></b>'
    else:
        return f'<gr><i>ms. </i></gr> <blue><b>{ortho_ms}</b></blue>; <gr><i>fs. </i></gr> <red><b>{ortho_fs}</red></b>'
        # raise Exception(f'Invalid pos passed to ms_fs_bold() for {lemme}')


def sp_bold(lemme, pos, ortho_s, ortho_p, genre=None) -> str:
    """
    :return: formatted string one singular and one plural
    """
    if pos == 'nom':
        if genre is None:
            raise Exception('Invalid arguments passed to sp_bold for where pos is nom')
        elif genre == 'm':
            c_male = apply_contraction(f'le {ortho_s}')
            return f'<b><blue>{c_male}</blue></b> [<gr><i>pl. </i></gr><blue>les {ortho_p}</blue>]'
        elif genre == 'f':
            c_fem = apply_contraction(f'la {ortho_s}')
            return f'<b><red>{c_fem}</red></b> [<gr><i>pl. </i></gr><red>les {ortho_p}</red>]'
        elif genre == 'ms_fp':
            c_male = apply_contraction(f'le {ortho_s}')
            return f'<b><blue>{c_male}</blue></b> [<gr><i>pl. </i></gr><red>les {ortho_p}</red>]'
        elif genre == 'fs_mp':
            c_fem = apply_contraction(f'la {ortho_s}')
            return f"<b><red>{c_fem}</red></b> [<gr><i>pl. </i></gr><blue>les {ortho_p}</blue>]"
        else:
            raise Exception('Invalid genre passed to sp_bold() where pos is nom')
    else:
        return f'<b>{lemme}</b> [<gr><i>pl. </i></gr>{ortho_p}]'


def mpf_det_bold(lemme, pos, ortho_m, ortho_p, ortho_f) -> str:
    """
    :return: formatted string one masculine singular, one plural, and one feminine singular
    """
    if pos == 'nom':
        # format: male / plural/ feminine (exists exclusively for nom)
        c_male = apply_contraction(f'le {ortho_m}')
        return f"<b><blue>{c_male}</blue></b> [<gr><i>pl. </i></gr><blue>les {ortho_p}</blue>; <gr><i>f. </i></gr><red>la {ortho_f}</red>]"
    else:
        return f"<b>{lemme}</b> [<gr><i>m. </i></gr><blue>{ortho_m}</blue>; <gr><i>pl. </i></gr><blue>{ortho_p}</blue>; <gr><i>f. </i></gr><red>{ortho_f}</red>]"


def four_bold(lemme, pos, ortho_ms, ortho_mpl, ortho_fs, ortho_fpl) -> (str|None):
    """
    :return: formatted string one masculine singular, one masculine plural, one feminine singular, and one feminine plural
    """
    # format noun four
    if pos == 'nom':
        return (
            f"<b>{lemme}</b> ["
            f"<gr><i>ms. </i></gr> <blue>le {ortho_ms}</blue>; "
            f"<gr><i>mpl. </i></gr> <blue>les {ortho_mpl}</blue>; "
            f"<gr><i>fs. </i></gr> <red>la {ortho_fs}</red>; "
            f"<gr><i>fpl. </i></gr> <red>les {ortho_fpl}</red>"
            "]"
        )
    elif pos == 'adj':
        return (
            f"<b>{lemme}</b> "
            f"[<gr><i>ms. </i></gr><blue>{ortho_ms}</blue>; "
            f"<gr><i>mpl. </i></gr><blue>{ortho_mpl}</blue>; "
            f"<gr><i>fs. </i></gr><red>{ortho_fs}</red>; "
            f"<gr><i>fpl. </i></gr><red>{ortho_fpl}</red>]"
        )
    else:
        return None


def find_row(rows, g, n):
    # returns the first row genre and nombre equal the inputs
    # None may be passed as NaN
    if g is None and n is None:
        r = rows[(rows['genre'].isna()) & (rows['nombre'].isna())]
        return r.iloc[0] if not r.empty else None
    elif g is None:
        r = rows[(rows['genre'].isna()) & (rows['nombre'] == n)]
        return r.iloc[0] if not r.empty else None
    elif n is None:
        r = rows[(rows['genre'] == g) & (rows['nombre'].isna())]
        return r.iloc[0] if not r.empty else None
    else:
        r = rows[(rows['genre'] == g) & (rows['nombre'] == n)]
        return r.iloc[0] if not r.empty else None


def apply_contraction(text) -> str:
    """
    :param text:
    :return: returns formatted article + word (ortho or lemme) w/ (f) if applicable
                for male: return le + word | l'word
                for fem:  return la + word | l'word (f)
    """
    def repl(m):
        article = m.group(1)
        word = m.group(2)
        vowels = 'aeiouhâàéèêëïîôùûü'
        if word and word[0].lower() in vowels and article == 'le':
            return f"l'{word}"
        elif word and word[0].lower() in vowels and article == 'la':
            return f"l'{word} (f)"
        else:
            return m.group(0)

    pattern = re.compile(r"\b(le|la|les) (\S+)")
    return pattern.sub(repl, text)


def print_formatting_exceptions(lemme, pos, lemme_df):
    print(f"Unhandled case for {lemme}, {pos}")
    for _, r in lemme_df.iterrows():
        ortho_val = r.get('ortho', 'NaN')
        genre_val = r.get('genre', 'NaN')
        nombre_val = r.get('nombre', 'NaN')
        print(f"\tortho: {ortho_val}, genre: {genre_val}, nombre: {nombre_val}")


def get_pronunciation(lemme, lemme_df):
    pronun_row = lemme_df[lemme_df['ortho'] == lemme]
    if not pronun_row.empty:
        return pronun_row['orthosyll'].iloc[0]
    else:
        return lemme_df['orthosyll'].iloc[0]


def update_to_export_rows(lemme, pos, noun_decls, pronunciation, translation, export_rows):
    if noun_decls is not False:
        if not isinstance(noun_decls, list):
            noun_decls = [noun_decls]

        for noun_decl in noun_decls:
            # apply contraction rule to noun_decl
            noun_decl = apply_contraction(noun_decl)

            # append
            export_rows.append({
                'Lemme': lemme,
                'Noun Declension': noun_decl,
                'Pronunciation': pronunciation,
                'Sound': '',
                'Translation': translation,
                'POS': pos,
                'Tags': '',
            })


def write_anki_csv(freq_start, chunk_idx, lemme_chunk, export_rows, formatting_exception_count):
    # output file name
    start_idx = freq_start + chunk_idx
    end_idx = start_idx + len(lemme_chunk) - 1
    out_file = os.path.join(
        OUTPUT_DIR, f'{OUTPUT_PREFIX}{start_idx}-{end_idx}.csv'
    )

    # create DataFrame for export - first row (makes it easier to import into Anki)
    export_df = pd.DataFrame(export_rows)

    # export CSV with UTF-8 and without index
    export_df.to_csv(out_file, index=False, header=False, encoding='utf-8')

    print(f'Exported {len(lemme_chunk)} lemmes to {out_file}')
    print(f'Formatting exceptions: {formatting_exception_count}')


if __name__ == "__main__":
    main()


"""
This comment contains the if-statement that makes sense. I used to some substitution plus
DeMorgan's Law to convert that into the monstrous if-statement you see below this comment.

This was necessary because we cannot check if a genre is "f" or 'm' without first checking
within its own local evaluated conditional that r1_genre is not NaN.

This 'simplification' allows us to handle empty excel cells without errors.

# if not ((r1_genre == 'm' and r2_genre == "f") or (r1_genre == "f" and r2_genre == 'm')):

if (((pd.isna(r1_genre) or r1_genre == "f") or (pd.isna(r2_genre) or r2_genre == "f")) and
        ((pd.isna(r1_genre) or r1_genre == 'm') or (pd.isna(r2_genre) or r2_genre == "f"))):
        
AHAHAHA HAH! We can just try: except: for NaN errors. Rip.
"""
