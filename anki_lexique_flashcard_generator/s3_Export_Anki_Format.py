"""
@Purpose: Format lexique to be anki-importable csvs.

@Instructions: s4 is designed to be run after this. You can run s3 any amount of times and
                then run s4 without any configuration and it will work just fine.

Design note 1: The formatting for adjectives and nouns is surprisingly inconvenient to fully decouple.
                I tried but the row functions are still messy. They work pretty well though, mostly
                through trial and error.

Design note 2: The Lexique 3.83 excel is usually (not always) sorted such that singular 's' rows
                come prior to plural 'p' rows even if the 's' and/or 'p' is missing. For that reason
                depending on what's missing, this code gambles that the pattern will hold true.

        Todo:
         - Finish implementing AWS Translate function and merging w/ DeepL in translate()... (my account service won't active (_(- -)_) so I can't test the dang thing)
         - Ugh, just remove the homo, homo, homo 1 -> 2 card code. It wasn't at all worth it.
"""
import copy
import os
import re
import time
from ast import literal_eval
from dataclasses import dataclass

# import boto3 # AWS library
import deepl
import numpy as np
import pandas as pd

# project files
from s1_Filter_Lexique import OUTPUT_DIR, DESIRED_FLASHCARDS
from s2_Mux_Lexique import mux_frequencies
from s2_Mux_Lexique import CHUNK_SIZE
from s8_DeNoise_Forvo_Audio import override_prog_configs_from_file


# ==== CONFIGURATION ====
# you probably don't need to change these
DECK_GROUPING_PREFIX = 'deckID_'    # you can rename this to whatever you want; it's just there to help you group decks/subdecks
PARSE_CUSTOM_DECK = False           # mostly here for the author, leave as false unless you want to rewrite parse_translations_from_exported_deck() to parse/carryover field values from one of your old decks
ANKI_CSV_OUTPUT_DIR = f'{OUTPUT_DIR}/anki_lexique_imports'
OUTPUT_PREFIX = 'anki_deck_'
OVERFLOW_FILENAME = f'df_overflow.csv'
OVERFLOW_PATH = f'{ANKI_CSV_OUTPUT_DIR}/{OVERFLOW_FILENAME}'
# ========================

# POS priority for sorting and filtering
POS_PRIORITY = ['adj', 'adv', 'prep', 'v', 'ono', 'n', 'con']


@dataclass
class Card:
    def __init__(self, lemme):
        self.lemme = lemme
        self.pos = ''
        self.pronunciation = ''
        self.ipa = ''
        self.noun_decl = ''
        self.translation = ''
        # for intermediate usage
        self.sing = ''
        self.plural = ''


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


formatting_exception_count = 0

def main():
    export_anki_format_csvs()


def export_anki_format_csvs(desired_flashcards=DESIRED_FLASHCARDS) -> None:
    print('Exporting Anki Formatted CSVs...')
    override_prog_configs_from_file(globals())

    # read custom deck (if configured)
    foreign_lemme_trans_pairs = {}
    if PARSE_CUSTOM_DECK:
        foreign_lemme_trans_pairs = parse_translations_from_exported_deck()

    # prep for DeepL translations
    deepl_auth_key = read_deepl_creds()
    deepl_client = deepl.DeepLClient(deepl_auth_key)
    deepl_source_language = 'FR'
    deepl_target_language = 'EN-US'

    # prep for AWS translations
    aws_access_key_id, aws_secret_access_key = read_aws_creds()
    # aws_client = boto3.client('translate', region_name='us-east-2', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key) # todo
    aws_client = ''
    aws_src_lang = 'en'
    aws_tgt_lang = 'fr'

    # get filtered, muxed, lexique lemmes and df
    muxed_lemmes, muxed_df = mux_frequencies()

    # region get starting lemme, index, filename
    start_lemme, deck_id, start_filename = get_start_lemme()
    mux_start_index = None

    if start_lemme == '':
        mux_start_index = 0
    elif start_lemme in muxed_lemmes:
        mux_start_index = muxed_lemmes.index(start_lemme) + 1 # +1 to start on the next card
    else:
        raise Exception(f'Lemme {start_lemme} from file {ANKI_CSV_OUTPUT_DIR} not present in top {desired_flashcards} desired muxed cards.')
    # endregion

    # region set desired number of flashcards
    if desired_flashcards == 0 or desired_flashcards > len(muxed_df):
        max_cards = len(muxed_df)
    else:
        max_cards = desired_flashcards
    # endregion

    # make flashcards
    for start_idx in range(mux_start_index, max_cards, CHUNK_SIZE):
        end_idx = start_idx + CHUNK_SIZE
        lemme_chunk = muxed_lemmes[start_idx: end_idx]
        deck_id += 1

        chunk_df = muxed_df[muxed_df['lemme'].isin(lemme_chunk)].copy()
        chunk_df.loc[:, '__order'] = pd.Categorical(chunk_df['lemme'], categories=lemme_chunk, ordered=True)
        chunk_df = chunk_df.sort_values('__order').drop(columns='__order')

        # writes flashcards to file
        end_filename = start_filename - 1 + CHUNK_SIZE
        print(f'Creating csv {start_filename} - {end_filename}.')
        create_flashcard_rows(lemme_chunk, chunk_df, deck_id, start_filename, end_filename, deepl_client, deepl_source_language, deepl_target_language, aws_client, aws_src_lang, aws_tgt_lang, foreign_lemme_trans_pairs)

        # increment filename
        start_filename += CHUNK_SIZE

    # close AWS client
    # aws_client.close()

    print('Exported CSVs.')
    print('')


def get_start_lemme() -> tuple[str, int, int]:
    def get_lemme_from_last_row_in_csv(filename):
        global DECK_GROUPING_PREFIX
        path = f'{ANKI_CSV_OUTPUT_DIR}/{filename}'
        df = pd.read_csv(path, encoding='utf-8', header=None)
        df_last_row = df.iloc[[-1]]
        lemme = str(df_last_row.values[0][0])
        deckID_s = str(df_last_row.values[0][7])
        deckID = int(deckID_s.replace(DECK_GROUPING_PREFIX, ''))
        return lemme, deckID

    highest_anki_file_path = ''
    if os.path.exists(ANKI_CSV_OUTPUT_DIR):
        dir_contents = os.listdir(ANKI_CSV_OUTPUT_DIR)
        if len(dir_contents) > 0:
            highest_anki_content_filename = ''
            highest_anki_card = 0
            lemme = ''
            deckID = 0
            for content in dir_contents:
                if 'lock' in content: # ignore temp file lock (such as if a csv is open in a excel/libreoffice)
                    continue
                else:
                    high = int(content.split('-')[-1].split('.')[0].strip())
                    if high > highest_anki_card:
                        highest_anki_content_filename = content
                        highest_anki_card = high

            if highest_anki_content_filename != '' and lemme == '':
                lemme, deckID = get_lemme_from_last_row_in_csv(highest_anki_content_filename)
            return lemme, deckID, highest_anki_card+1

    return '', 0, 1


def parse_translations_from_exported_deck() -> dict:
    """
    You can pretty much ignore this function. It parses the french deck that I studied
     previously. I made a lot of translation edits/hints that I wanted to carry forward.

     If you want to use it you'll probably have to rewrite from the ground up although honestly
     it's probably easier than you think.
    :return: returns dict of lemme/translation pairs.
    """
    fields = [
        'Word',                     # this is the deck's lemme
        'Word with article',
        'Frequency Index',
        'IPA',
        'Noun declension',
        'Basic meanings of word',   # this is the deck's english translation (example: 'adj. excessive')
        'Example sentences',
        'Example sentences without translation',
        'Wiktionary entry',
        'Word with declinations',
        'Parisian French Audio (Voice 1)',
        'Canadian French Audio (Voice 1)'
    ]
    field_separator = '\t'
    starts_on_line_index = 2
    exported_deck_path = '../resources/1. 5000 Most Common French Words (Canadian)__English to French.txt'

    with open(exported_deck_path, 'r') as f:
        lines = f.readlines()
        data_lines = lines[starts_on_line_index:]

        foreign_lemme_list = []
        foreign_lemme_translation_lookup = {}
        for data_line in data_lines:
            field_data = data_line.split(field_separator)
            foreign_lemme = field_data[0].strip().lower()

            # skip potential duplicates - that's just asking for trouble
            if foreign_lemme not in foreign_lemme_list:
                # format translation (substitude to prevent unwanted splitting on delimeter)
                foreign_translation_field = field_data[5].strip()
                sub1 = foreign_translation_field.replace('etc.', '(_+ _ +_/')   # substitute 'etc.'
                sub2 = sub1.replace('...', '(_^ _ ^_/')    # substitute '...'
                foreign_translation_s2 = sub2.split('.')[-1].strip().lower() # split on '.'
                foreign_translation_s1 = foreign_translation_s2.replace('(_+ _ +_/', 'etc.') # sub 'etc.'
                foreign_translation = foreign_translation_s1.replace('(_^ _ ^_/', '...')  # sub '...'

                # when I started I would do hints as english, let's correct that inconsistency
                foreign_translation = foreign_translation.replace('(not ', '(pas ')

                # append
                foreign_lemme_list.append(foreign_lemme)
                foreign_lemme_translation_lookup[foreign_lemme] = foreign_translation

        return foreign_lemme_translation_lookup


def create_flashcard_rows(lemmes, df, deck_id, start_filename, end_filename,
                          deepl_client, deepl_src_lang, deepl_tgt_lang,
                          aws_client, aws_src_lang, aws_tgt_lang,
                          translation_pairs):
    # ensure output directory exists
    os.makedirs(ANKI_CSV_OUTPUT_DIR, exist_ok=True)

    # create a set of {"lemme1":[row_A, row_B], "lemme2":[row_X,...]} & ["lemme1", "lemme2", ...]
    lemmes, lemme_to_rows = group_rows_by_lemme(df)

    # populate and write export_rows
    export_rows = []
    card_queue = []         # to be done in batches
    translation_queue = {}  # to be done in batches; structure {'lemme': card_needing_translation, ...}
    MAX_BATCH_SIZE = 50 # for me this is the DeepL texts limitation, if anyone ever actually looks at this then it'll be the smaller of your AWS/DeepL limitation
    for i_lemme, lemme in enumerate(lemmes):
        lemme_df = pd.DataFrame(lemme_to_rows[lemme])
        card = Card(lemme)

        # pos
        card.pos = lemme_df['cgram'].iloc[0]

        # copy orthosyll column to pronunciation column - use matching lemme orthosyll if available
        card.pronunciation = get_pronunciation(lemme, lemme_df)

        # ipa
        card.ipa = map_sampa_to_ipa(lemme, lemme_df)

        # format 'Noun Declension' field
        tmp_noun_decl = format_noun_declension(lemme_df, card_queue, card, deepl_client, deepl_src_lang, deepl_tgt_lang, translation_pairs)
        if tmp_noun_decl is not None:
            card.noun_decl = tmp_noun_decl
            if card.lemme in translation_pairs.keys():
                card.translation = translation_pairs[lemme]
            else:
                translation_queue[card.lemme] = card
            card_queue.append(card)

        # assign 'English Translation'
        if (i_lemme+1) % MAX_BATCH_SIZE == 0 or len(card_queue) == CHUNK_SIZE: # +1 for index, 50 is batch size; do batch translation when full
            # batch update card translation fields (if needed)
            if len(list(translation_queue.keys())) > 0:
                deepl_batch_translate(deepl_client, translation_queue, card_queue, deepl_src_lang, deepl_tgt_lang, translation_pairs)
                translation_queue = {} # reset translation queue

            # batch add card queue to export_rows (ergo card queue no longer needed)
            for card in card_queue:
                update_export_rows(card, deck_id, export_rows)
            card_queue = [] # reset card queue
            print(f'  Batch up to lemme {i_lemme+1} in memory.')

    # write sheet to be imported into Anki
    write_anki_csv(start_filename, end_filename, export_rows)


def read_aws_creds():
    """
    :return: AWS credentials
    """
    abs_path = os.path.abspath(os.getcwd())
    credentials_file = '../resources/aws_credentials.txt'

    # ensure file exists, make file template if it doesn't
    if not os.path.exists(credentials_file):
        with open(credentials_file, 'w+') as f:
            cred_template = ("{\n"
                             "\t'aws_access_key_id': 'PASTE_HERE__leave_quotes',\n"
                             "\t'aws_secret_access_key': 'PASTE_HERE__leave_quotes'\n"
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
            aws_access_key_id = cred_dict['aws_access_key_id']
            aws_secret_access_key = cred_dict['aws_secret_access_key']
            return aws_access_key_id, aws_secret_access_key
        except SyntaxError or ValueError as e:
            print(f'\nCredential file at {abs_path}/{credentials_file} is malformed. Please check your credentials file.\nNote: if you delete your file and re-run this program it will remake the template. After, copy/paste your credentials where indicated.')
            exit(-1)


def read_deepl_creds():
    """
    :return: DeepL API key
    """
    abs_path = os.path.abspath(os.getcwd())
    credentials_file = '../resources/deepl_credentials.txt'

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


def deepl_batch_translate(deepl_client, translation_queue, cards, deepl_src_lang, deepl_tgt_lang, exported_translation_pairs):
    num_attempts = 0
    delay = 1
    translation_keys = list(translation_queue.keys())
    while True:
        time.sleep(delay)
        try:
            responses = deepl_client.translate_text(translation_keys, source_lang=deepl_src_lang, target_lang=deepl_tgt_lang)
            # number of translation responses should equal number of items in queue
            if len(responses) != len(translation_queue):
                raise Exception('Invalid DeepL response. Source and translation pairs do not match.')

            for i in range(0, len(responses)):
                response = responses[i]
                translation = response.text
                lemme = translation_keys[i]
                card = translation_queue[lemme]

                # handle inconsistent prescence of 'to ' prior to verbs (and make lowercase)
                card.translation = standardize_translation(card.pos, translation)

            break
        except Exception as e:
            num_attempts += 1
            if num_attempts > 5:
                raise
            delay *= 1.5


def translate(lemme, pos, deepl_client, deepl_src_lang, deepl_tgt_lang, exported_translation_pairs) -> (str|None):
    """
    :return: returns target language translation(s) as str, or None if no translations found.
    """
    # prefer local translation - note: this is skipped unless flag is toggled on
    if lemme in exported_translation_pairs.keys():
        return exported_translation_pairs[lemme]

    # try DeepL & AWS
    deepl_translation = deepl_translate(lemme, pos, deepl_client, deepl_src_lang, deepl_tgt_lang)
    aws_translation = ''
    # aws_translation = aws_translate(aws_client, lemme, aws_src_lang, aws_tgt_lang)

    # handle inconsistent prescence of 'to ' prior to verbs (and make lowercase)
    deepl_translation = standardize_translation(pos, deepl_translation)
    aws_translation = standardize_translation(pos, aws_translation)

    # combine translation as reasonable
    if len(deepl_translation) == 0 and len(aws_translation) == 0:
        return None
    elif len(deepl_translation) == 0:
        return aws_translation
    elif len(aws_translation) == 0:
        return deepl_translation
    else:
        if deepl_translation == aws_translation:
            return deepl_translation
        else:
            return f'{deepl_translation}; {aws_translation}'


def standardize_translation(pos, translation):
    if pos == 'v':
        if len(translation) > 0:
            if 'to ' not in translation:
                # prepend 'to '
                translation = f'to {translation}'
    return translation.lower()


def deepl_translate(lemme, pos, deepl_client, source_lang, target_language, max_attempts=5) -> str:
    """
    :return: DeepL translation
    """
    delay = 1  # seconds
    deepl_translation = ''
    for attempt in range(1, max_attempts + 1):
        try:
            deepl_translation = deepl_client.translate_text(lemme, source_lang=source_lang, target_lang=target_language).text
        except deepl.exceptions.AuthorizationException as e:
            print('')
            try:
                raise Exception(e)
            finally:
                print('')
                print('DeepL API authorization failure. Ensure correct key from DeepL website (hint: find under "Account->API Keys and Limits") is pasted in resources/deepl_credentials.txt')
        except deepl.exceptions.TooManyRequestsException as e:
            if attempt == max_attempts:
                raise Exception(e)
            print(f'429 Too Many Requests — retrying in {delay}s (attempt {attempt})')
            time.sleep(delay)
            delay *= 2  # exponential backoff

    return deepl_translation


def TODO_NOT_FINISHED_aws_translate(client, text, src_lang, tgt_lang):
    """
    :return: AWS translation
    other translate calls of note:
      list_text_translation_jobs list async batch https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/translate/client/list_text_translation_jobs.html
      start_text_translation_job start async batch https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/translate/client/start_text_translation_job.html
      stop_text_translation_job  stop async batch
    """
    response = client.translate_text(
        Text=f'{text}',
        TerminologyNames=[
            '', # not wanted?
        ],
        SourceLanguageCode=f'{src_lang}',
        TargetLanguageCode=f'{tgt_lang}',
        Settings={
            'Formality': '', # values 'FORMAL' or 'INFORMAL'
            'Profanity': '', # values 'MASK' or ?
            'Brevity': 'OFF' # values 'ON' or ?
        }
    )

    return response.get('TranslatedText').lower() # from https://docs.aws.amazon.com/translate/latest/dg/get-started-sdk.html
    # return response['TranslatedText'].lower() # is this the right return format, hard to know w/out an API key to test...


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


def format_noun_declension(rows, card_queue, card, deepl_client, deepl_src_lang, deepl_tgt_lang, translation_pairs):
    """
    :return: return formatted html as str or return None to continue to next lemme.
    """
    global formatting_exception_count
    lemme = card.lemme
    pos = card.pos

    # we'll just pre-compute this junk. it's a little inefficient but it reduces code complexity
    row_stats = Row_Stats(rows=rows)

    # check for hard-coded exceptions first
    noun_decl = handle_hard_coded_formats(rows, lemme)
    if noun_decl is not None:
        if noun_decl is False:
            return None
        return noun_decl

    if pos in {'v', 'adv', 'prep', 'con', 'ono'}:
        noun_decl = singular_bold(lemme, pos)
    else:
        num_rows = len(rows)
        try:
            # if 1 row OR genre empty and all ortho's are equal then treat as single
            if num_rows == 1 or (all(x == lemme for x in rows['ortho']) and rows['genre'].isna().all()):
                noun_decl = row1_func(rows, row_stats, lemme, pos)
            elif num_rows == 2:
                noun_decl = row2_func(rows, row_stats, lemme, pos, card)
            elif num_rows == 3:
                noun_decl = row3_func(rows, row_stats, lemme, pos)
            elif num_rows == 4:
                noun_decl = row4_func(rows, row_stats, lemme, pos)
            elif num_rows == 5:
                noun_decl = row5_func(rows, row_stats, lemme, pos)
            else:
                noun_decl = None
        except ValueError or TypeError:
            noun_decl = None

    # if formatting fails, print info. so we can try to fix the lexique
    if noun_decl is None:
        formatting_exception_count += 1
        print_formatting_exceptions(lemme, pos, rows)

        # just treat lemme as singular adj - e.g. give up and settle for bolding the lemme
        noun_decl = bold_wrapper(lemme)
    elif isinstance(noun_decl, list):  # note: can only occur from pos is noun
        # assign 'English Translation'
        male_translation = translate(f'le {lemme}', pos, deepl_client, deepl_src_lang, deepl_tgt_lang,
                                     translation_pairs).replace('the ', '')
        # fem_translation = translate(f'la {lemme}', pos, deepl_client, deepl_src_lang, deepl_tgt_lang,
        #                             translation_pairs).replace('the ', '')

        # if male_translation != fem_translation:
        #     card_m = card
        #     card_f = copy.copy(card)
        #     card_m.translation = male_translation
        #     card_f.translation = fem_translation
        #     card_m.noun_decl = noun_decl[0]
        #     card_f.noun_decl = noun_decl[1]
        #     # do NOT add to translation queue - already translated!
        #     card_queue.append(card_m)
        #     card_queue.append(card_f)
        #     return None  # move to next lemme
        # else:
            # correct false positive homo, homo, same lemme diff meaning row_2() conditional -> mpf (probably)
            # _ s
            # _ p
        card.noun_decl = mpf_det_bold(lemme, 'n', card.sing, card.plural, card.sing)
        card.translation = male_translation
        # do NOT add to translation queue - already translated!
        card_queue.append(card)
        return None # move to next lemme

    return noun_decl


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


def row2_func(rows, rs, lemme, pos, card):
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

    # 'adj' or 'n'
    if rs.num_m_s == 1 and rs.num_f_s == 1:
        #   genre   nombre  |   genre   nombre
        #     m       s     |     f       s
        #     f       s     |     s       s
        return ms_fs_bold(lemme, pos, rs.rows_m_s.iloc[0]['ortho'], rs.rows_f_s.iloc[0]['ortho'])

    # nom
    if pos == 'n':
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
            card.sing = ortho_s
            card.plural = ortho_p
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

    if pos == 'n' or pos == 'adj':
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
    if pos == 'n':
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
    if pos == 'adj':
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
    if pos == 'n':
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
            ms = rs.rows_m_s
            mpl = rs.rows_m_p
            fs = rs.rows_f_s
            fpl = rs.rows_f_p

            if rs.num_na_s == 1 and rs.num_na_p == 1 and rs.num_f_s == 1 and rs.num_f_p == 1:
                # works for 'adj' and 'n' - assumes male is more likely than two archaics
                #   genre   nombre
                #     _       s     ->  m   s
                #     _       p     ->  m   p
                #     f       s
                #     f       p
                ortho_ms = rs.rows_na_s.iloc[0]['ortho']
                ortho_mp = rs.rows_na_p.iloc[0]['ortho']
                return four_bold(lemme, pos, ortho_ms, ortho_mp,
                                 rs.rows_f_s.iloc[0]['ortho'], rs.rows_f_p.iloc[0]['ortho'])

            if not ms.empty and not mpl.empty and fs.empty and fpl.empty:
                # nothing missing
                return four_bold(lemme, pos, ms.iloc[0]['ortho'], mpl.iloc[0]['ortho'], fs.iloc[0]['ortho'], mpl.iloc[0]['ortho'])
            else:
                if lemme == 'lapon':
                    pass
                # one row missing
                if (not ms.empty and not mpl.empty and not fs.empty) or (not ms.empty and not mpl.empty and not fpl.empty) or (not ms.empty and not fs.empty and not fpl.empty) or (not mpl.empty and not fs.empty and not fpl.empty):
                    # assign ms|mpl|fs|fpl to row with missing genre/nombre malformed - process of elimination
                    malformed_row = rows[rows['genre'].isna() | rows['nombre'].isna()]
                    if not ms.empty and not mpl.empty and not fs.empty:
                        fpl = malformed_row
                        if malformed_row.empty:
                            fpl = mpl
                    elif not ms.empty and not mpl.empty and not fpl.empty:
                        fs = malformed_row
                    elif not ms.empty and not fs.empty and not fpl.empty:
                        mpl = malformed_row
                        if malformed_row.empty:
                            mpl = fpl
                    else:
                        ms = malformed_row

                    # return corrected value
                    ortho_ms = ms.iloc[0]['ortho']
                    ortho_mpl = mpl.iloc[0]['ortho']
                    ortho_fs = fs.iloc[0]['ortho']
                    ortho_fpl = mpl.iloc[0]['ortho']
                    return four_bold(lemme, pos, ortho_ms, ortho_mpl, ortho_fs, ortho_fpl)
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
    :return: return formatted str for formatting exceptions not easily filtered/prior or corrected dynamically
    note: returning false prevents duplicates from getting exported.
            DO NOT CHANGE False TO None - it's used to prevent appending flashcards to export_rows.
    """
    # region exceptions because of duplicate lemme entries in lexique that should be consolidated into one card
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
    # endregion

    # region exception because singular ortho is missing from lexique
    if lemme == 'fois':
        return sp_bold(lemme, 'n', lemme, lemme, 'f')
    # endregion

    # region exceptions for unique/archaic/rare poetic spellings (that can't be corrected easily w/ rules)
    if lemme == 'oeil':
        return sp_bold(lemme, 'n', lemme, 'yeux', 'm')
    elif lemme == 'lieu':
        return sp_bold(lemme, 'n', lemme, 'lieux', 'm')
    elif lemme == 'aïeul':
        ms_text = 'le aïeul'
        literal_mpl_text = 'les aïeuls'
        figurative_mpl_text = 'les aïeux'
        fs_text = 'la aïeule'
        return f'{bold_wrapper(lemme)} [<span class=gn><i>ms. </i></span> {span_wrapper(text=ms_text, is_bold=False, genre='m')}; <span class=gn><i>mpl. (refers to male members of a genealogical tree - literal grandfathers/forefathers)</i></span> {span_wrapper(text=literal_mpl_text, is_bold=False, genre='m')}; <span class=gn><i>mpl. (refers to collective ancestors regardless of gender even if not from a single literal bloodline)</i></span> {span_wrapper(text=figurative_mpl_text, is_bold=False, genre='m')}; <span class=gn><i>fs. </i></span> {span_wrapper(text=fs_text, is_bold=False, genre='f')}; <span class=gn><i>fpl. </i></span> <i>ommitted - archaic</i>]'
    # endregion

    return None


def span_wrapper(text, is_bold, genre):
    if is_bold:
        text = bold_wrapper(text)

    if genre == 'm':
        return f'<span class="masc">{text}</span>'
    elif genre == 'f':
        return f'<span class="fem">{text}</span>'
    else:
        raise Exception('Invalid genre, must be "m" or "f".')


def bold_wrapper(text):
    return f'<b>{text}</b>'


def singular_bold(ortho_s, pos, genre=None) -> str:
    """
    :return: formatted string for one singular
    """
    if pos == 'n':
        if genre is None:
            raise Exception('Invalid arguments passed to singular_bold() for where pos is nom')
        elif genre == 'm':
            c_male = apply_article_elision(f'le {ortho_s}')
            return f'{span_wrapper(text=c_male, is_bold=True, genre='m')}'
        elif genre == 'f':
            c_fem = apply_article_elision(f'la {ortho_s}')
            return f'{span_wrapper(text=c_fem, is_bold=True, genre='f')}'
        else:
            raise Exception('Invalid genre passed to to det function: singular_bold()')
    else:
        return f'{bold_wrapper(ortho_s)}'


def plural_bold(plural_ortho, pos, genre=None) -> str:
    """
    :return: formatted string for one plural
    """
    plural_text = f'les {plural_ortho}'
    if pos == 'n':
        if genre == 'm':
            return f'{span_wrapper(text=plural_text, is_bold=True, genre='m')}'
        elif genre == 'f':
            return f'{span_wrapper(text=plural_text, is_bold=True, genre='f')}'
        else:
            # this might should be an exception... meh, I'll allow it
            return f'{bold_wrapper(text=plural_text)}'
    else:
        return f'{bold_wrapper(text=plural_ortho)}'


def ms_fs_bold(lemme, pos, ortho_ms, ortho_fs) -> str:
    """
    :return: formatted one masculine singular and one feminine singular
    """
    # shout [grand-papa, grand-mama]
    if pos == 'n':
        c_male = apply_article_elision(f'le {ortho_ms}')
        fs_text = f'la {ortho_fs}'
        return f'<span class=gn><i>ms. </i></span> {span_wrapper(text=c_male, is_bold=True, genre='m')}; <span class=gn><i>fs. </i></span> {span_wrapper(text=fs_text, is_bold=True, genre='f')}'
    else:
        return f'<span class=gn><i>ms. </i></span> {span_wrapper(text=ortho_ms, is_bold=True, genre='m')}; <span class=gn><i>fs. </i></span> {span_wrapper(text=ortho_fs, is_bold=True, genre='f')}'
        # raise Exception(f'Invalid pos passed to ms_fs_bold() for {lemme}') ??


def sp_bold(lemme, pos, ortho_s, ortho_p, genre=None) -> str:
    """
    :return: formatted string one singular and one plural
    """
    if pos == 'n':
        plural_text = f'les {ortho_p}'
        if genre is None:
            raise Exception('Invalid arguments passed to sp_bold for where pos is nom')
        elif genre == 'm':
            c_male = apply_article_elision(f'le {ortho_s}')
            return f'{span_wrapper(text=c_male, is_bold=True, genre='m')} [<span class=gn><i>pl. </i></span>{span_wrapper(text=plural_text, is_bold=False, genre='m')}]'
        elif genre == 'f':
            c_fem = apply_article_elision(f'la {ortho_s}')
            return f'{span_wrapper(text=c_fem, is_bold=True, genre='f')} [<span class=gn><i>pl. </i></span>{span_wrapper(text=plural_text, is_bold=False, genre='f')}]'
        elif genre == 'ms_fp':
            c_male = apply_article_elision(f'le {ortho_s}')
            return f'{span_wrapper(text=c_male, is_bold=True, genre='m')} [<span class=gn><i>pl. </i></span>{span_wrapper(text=plural_text, is_bold=False, genre='f')}]'
        elif genre == 'fs_mp':
            c_fem = apply_article_elision(f'la {ortho_s}')
            return f"{span_wrapper(text=c_fem, is_bold=True, genre='f')} [<span class=gn><i>pl. </i></span>{span_wrapper(text=plural_text, is_bold=False, genre='m')}]"
        else:
            raise Exception('Invalid genre passed to sp_bold() where pos is nom')
    else:
        return f'{bold_wrapper(text=lemme)} [<span class=gn><i>pl. </i></span>{ortho_p}]'


def mpf_det_bold(lemme, pos, ortho_m, ortho_p, ortho_f) -> str:
    """
    :return: formatted string one masculine singular, one plural, and one feminine singular
    """
    if pos == 'n':
        # format: male / plural/ feminine (exists exclusively for nom)
        c_male = apply_article_elision(f'le {ortho_m}') # ms_text conjuction
        plural_text = f'les {ortho_p}'
        fs_text = f'la {ortho_f}'
        return f'{span_wrapper(text=c_male, is_bold=True, genre='m')} [<span class=gn><i>pl. </i></span>{span_wrapper(text=plural_text, is_bold=False, genre='m')}; <span class=gn><i>f. </i></span>{span_wrapper(text=fs_text, is_bold=False, genre='f')}]'
    else:
        return f'{bold_wrapper(text=lemme)} [<span class=gn><i>m. </i></span>{span_wrapper(text=ortho_m, is_bold=False, genre='m')}; <span class=gn><i>pl. </i></span>{span_wrapper(text=ortho_p, is_bold=False, genre='m')}; <span class=gn><i>f. </i></span>{span_wrapper(text=ortho_f, is_bold=False, genre='f')}]'


def four_bold(lemme, pos, ortho_ms, ortho_mpl, ortho_fs, ortho_fpl) -> (str|None):
    """
    :return: formatted string one masculine singular, one masculine plural, one feminine singular, and one feminine plural
    """
    # format noun four
    if pos == 'n':
        ms_text = f'le {ortho_ms}'
        mpl_text = f'les {ortho_mpl}'
        fs_text = f'la {ortho_fs}'
        fpl_text = f'les {ortho_fpl}'
        return (
            f'{bold_wrapper(text=lemme)} ['
            f'<span class=gn><i>ms. </i></span> {span_wrapper(text=ms_text, is_bold=False, genre='m')}; '
            f'<span class=gn><i>mpl. </i></span> {span_wrapper(text=mpl_text, is_bold=False, genre='m')}; '
            f'<span class=gn><i>fs. </i></span> {span_wrapper(text=fs_text, is_bold=False, genre='f')}; '
            f'<span class=gn><i>fpl. </i></span> {span_wrapper(text=fpl_text, is_bold=False, genre='f')}'
            ']'
        )
    elif pos == 'adj':
        return (
            f'{bold_wrapper(text=lemme)} '
            f'[<span class=gn><i>ms. </i></span>{span_wrapper(text=ortho_ms, is_bold=False, genre='m')}; '
            f'<span class=gn><i>mpl. </i></span>{span_wrapper(text=ortho_mpl, is_bold=False, genre='m')}; '
            f'<span class=gn><i>fs. </i></span>{span_wrapper(text=ortho_fs, is_bold=False, genre='f')}; '
            f'<span class=gn><i>fpl. </i></span>{span_wrapper(text=ortho_fpl, is_bold=False, genre='f')}]'
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


def apply_article_elision(text) -> str:
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


def update_export_rows(card, deck_id, export_rows):
    if card.translation is not None:
        # append
        export_rows.append({
            'Lemme': card.lemme,
            'Noun Declension': card.noun_decl,
            'Pronunciation': card.pronunciation,
            'IPA': card.ipa,
            'Sound': '',
            'Translation': card.translation,
            'POS': card.pos,
            'Deck Id': f'{DECK_GROUPING_PREFIX}{deck_id}',
            'Tags': '',
        })


def write_anki_csv(start_idx, end_idx, export_rows):
    # output file name
    out_file = os.path.join(
        ANKI_CSV_OUTPUT_DIR, f'{OUTPUT_PREFIX}{start_idx} - {end_idx}.csv'
    )

    # create DataFrame for export - first row (makes it easier to import into Anki)
    export_df = pd.DataFrame(export_rows)

    # export CSV with UTF-8 and without index
    export_df.to_csv(out_file, index=False, header=False, encoding='utf-8')

    print(f'Exported {len(export_df)} lemmas to {out_file}')
    print(f'Formatting exceptions: {formatting_exception_count}')
    print('')


def map_sampa_to_ipa(lemme, lemme_df):
    """Convert a French SAMPA string (Lexique 3) to IPA"""
    sampa_row = lemme_df[lemme_df['ortho'] == lemme]
    if not sampa_row.empty:
        sampa = sampa_row['sampa'].iloc[0]
    else:
        sampa = lemme_df['sampa'].iloc[0]


    # ordered list: 2-char tokens first, then 1-char
    sampa_ipa_map = {
        'O~': 'ɔ̃', 'E~': 'ɛ̃', 'A~': 'ɑ̃', '9~': 'œ̃',
        'tS': 'tʃ', 'dZ': 'dʒ', 'gZ': 'gʒ',
        'S': 'ʃ', 'Z': 'ʒ', 'N': 'ŋ', 'R': 'ʁ', 'j': 'j', 'w': 'w', 'H': 'ɥ',
        '1': 'ʊ̃', '2': 'ø', '5': 'ɘ', '8': 'ø', '9': 'œ', '@': 'ə',
        'a': 'a', 'b': 'b', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g',
        'i': 'i', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o',
        'p': 'p', 's': 's', 't': 't', 'u': 'u', 'v': 'v', 'x': 'ks',
        'y': 'y', 'z': 'z',
        '~': '̃', '°': 'ə', '§': '',
        '-': '.',
    }

    def map_accented_to_ipa(text):
        ACCENTED_CHAR_MAP = {
            'é': 'e',  # /e/ - close-mid front unrounded
            'è': 'E',  # /ɛ/
            'ê': 'E',
            'ë': 'E',
            'à': 'a',
            'â': 'a',
            'î': 'i',
            'ï': 'i',
            'ô': 'o',
            'ù': 'y',
            'û': 'y',
            'ü': 'y',
            'ç': 's',
            'œ': '9',  # approximate
            'æ': 'a',  # close enough
            'É': 'e',
            'È': 'E',
            'Ê': 'E',
            'À': 'a',
            'Â': 'a',
            'Î': 'i',
            'Ï': 'i',
            'Ô': 'o',
            'Ù': 'y',
            'Û': 'y',
            'Ü': 'y',
            'Ç': 's',
            'Œ': '9',
            'Æ': 'a',
        }
        return ''.join(ACCENTED_CHAR_MAP.get(c, c) for c in text)

    # remove spaces and accents
    sampa = sampa.replace(' ', '')

    # region build ipa
    ipa = ''
    i = 0
    while i < len(sampa):
        # try 2-character match first
        if i + 1 < len(sampa) and sampa[i:i+2] in sampa_ipa_map:
            ipa += sampa_ipa_map[sampa[i:i+2]]
            i += 2
        elif sampa[i] in sampa_ipa_map:
            ipa += sampa_ipa_map[sampa[i]]
            i += 1
        else:
            ipa += sampa[i]  # fallback
            i += 1

    def add_ipa_css_spans(ipa):
        # add first/last css markup classes
        split_ipa = ipa.split('.')
        if len(split_ipa) == 1:
            split_ipa[0] = f'<span class="ipa_col_syl">{split_ipa[0]}</span>'
        # elif len(split_ipa) == 2: enable for og formatting
        #     split_ipa[-1] = f'<span class="ipa_col_syl">{split_ipa[-1]}</span>'
        elif len(split_ipa) >= 2:
            split_ipa[0] = f'<span class="ipa_col_syl">{split_ipa[0]}</span>'
            split_ipa[-1] = f'<span class="ipa_col_syl">{split_ipa[-1]}</span>'
        ipa = '.'.join(split_ipa)

        # add innermost markup for periods
        ipa = ipa.replace('.', '<span class="ipa_period">.</span>')

        return ipa

    # add span classes for css coloring (for use in Styling in Anki)
    ipa = add_ipa_css_spans(ipa)

    return ipa.lower()


if __name__ == "__main__":
    main()
