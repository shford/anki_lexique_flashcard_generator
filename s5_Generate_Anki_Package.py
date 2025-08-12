import os

import pandas as pd
import random

import genanki

from s2_Mux_Lexique import DESIRED_FLASHCARDS, CHUNK_SIZE
from s4_Enforce_500_Anki_Cards import ANKI_CSV_DIR
from s4_Enforce_500_Anki_Cards import OUTPUT_PREFIX as ANKI_CSV_PREFIX

APKG_DIR = f'default_output/apkg_output'


def main():
    # init uids counter (to make each seed unique)
    uid_counter = [0]

    # make .apkg output dir
    os.makedirs(APKG_DIR, exist_ok=True)

    # define notetypes
    custom_french_forvo_model = define_custom_french_forvo_model(uid_counter)

    # init constants to create deck
    l2_counter = 0
    l2_increment = 4000
    l2_iterations = int(DESIRED_FLASHCARDS / l2_increment)
    l2_stop = l2_increment * l2_iterations

    # l3_counter = 0
    l3_stop = l2_increment
    # format l1 name/assign uid/ create deck
    l1_deck_name = 'French Forvo Deck'
    # print(l1_deck_name)
    l1_deck_uid = generate_inc_uids(uid_counter)
    l1_deck = genanki.Deck(deck_id=l1_deck_uid, name=l1_deck_name)
    total_l3_deck_counter = 1
    for l2_curr_start in range(1, l2_stop, l2_increment):
        # format l2 (child) deck name
        l2_counter += 1
        l2_curr_end = l2_curr_start + l2_increment - 1
        l2_subdeck_name = f'{l2_counter}. Most Comon French Words {l2_curr_start} - {l2_curr_end}'
        l2_deck_name = f'{l1_deck_name}::{l2_subdeck_name}'
        # print(f'\t{l2_deck_name}')

        l3_deck_counter_per_l2_loop = 1
        for l3_curr_start in range(l2_curr_start, l2_curr_start + l2_increment, CHUNK_SIZE):
            l3_label = chr(ord('a') + l3_deck_counter_per_l2_loop - 1)
            l3_curr_end = l3_curr_start + CHUNK_SIZE - 1
            l3_subdeck_name = f'{l3_label}. Freq {l3_curr_start} - {l3_curr_end}'
            l3_deck_name = f'{l2_deck_name}::{l3_subdeck_name}'
            l3_deck_counter_per_l2_loop += 1
            # print(f'\t\t{l3_deck_name}')

            # populate l3 deck (w/ notes)
            l3_deck_uid = generate_inc_uids(uid_counter)
            l3_deck = genanki.Deck(deck_id=l3_deck_uid, name=l3_deck_name)
            csv_name = f'{ANKI_CSV_PREFIX}{l3_curr_start} - {l3_curr_end}.csv'
            add_notes_to_deck_from_csv(l3_deck, csv_name, custom_french_forvo_model)

            # save each deck as .apkg
            package = genanki.Package(l3_deck)
            package.write_to_file(f'{APKG_DIR}/deck_{total_l3_deck_counter}.apkg')
            total_l3_deck_counter += 1

            en_to_fr_subdeck_name = 'English to French'
            en_to_fr_name = f'{l3_deck_name}::{en_to_fr_subdeck_name}'

            fr_to_en_subdeck_name = 'French to English'
            fr_to_en_name = f'{l3_deck_name}::{en_to_fr_subdeck_name}'

    return


def generate_inc_uids(unique: list):
    """
    Generates a random uid for Anki classes (decks/models/etc - can produce note guid if you override).
    :param unique: unique is a list containing an int that should probably just a counter
    :return:
    """
    static_seed = 1741382657 # should remain constant for stability if regenerating
    seeded_generator = random.Random(static_seed + unique[0])
    unique[0] += 1
    return seeded_generator.randint(1 << 30, (1 << 31) - 1)


def define_custom_french_forvo_model(uid_counter):
    ft_en_fr_path = 'resources/Anki Card Formats/Front Template__En_Fr.txt'
    with open(ft_en_fr_path, 'r') as f:
        ft_en_fr = f.read()

    bt_en_fr_path = 'resources/Anki Card Formats/Back Template__En_Fr.txt'
    with open(bt_en_fr_path, 'r') as f:
        bt_en_fr = f.read()

    ft_fr_en_path = 'resources/Anki Card Formats/Front Template__Fr_En.txt'
    with open(ft_fr_en_path, 'r') as f:
        ft_fr_en = f.read()

    bt_fr_en_path = 'resources/Anki Card Formats/Back Template__Fr_En.txt'
    with open(bt_fr_en_path, 'r') as f:
        bt_fr_en = f.read()

    # css but Anki calls it Styling (for Notes) so.. I will too
    styling_path_path = 'resources/Anki Card Formats/Styling.txt'
    with open(styling_path_path, 'r') as f:
        styling = f.read()

    model_uid = generate_inc_uids(uid_counter)
    custom_french_forvo_model = genanki.Model(
        model_id=model_uid,
        name='Custom French Forvo',
        fields=[
            {'name': 'Lemma'},
            {'name': 'Noun_Declension'},
            {'name': 'Pronunciation'},
            {'name': 'IPA'},
            {'name': 'Sound'},
            {'name': 'English_Meaning'},
            {'name': 'POS'},
            {'name': 'Deck_Id'},
        ],
        templates=[
            {
                'name': 'En_Fr',
                'qfmt': ft_en_fr,
                'afmt': bt_en_fr,
            },
            {
                'name': 'Fr_En',
                'qfmt': ft_fr_en,
                'afmt': bt_fr_en,
            },
        ],
        css=styling,
    )

    return custom_french_forvo_model


def add_notes_to_deck_from_csv(deck, csv_name, custom_french_forvo_model):
    # read csv
    path = f'{ANKI_CSV_DIR}/{csv_name}'
    df = pd.read_csv(path, encoding='utf-8', header=None)

    # populate notes
    for row_i in range(0, len(df)):
        # extract fields from cells
        row = df.iloc[row_i]
        lemme = row[0]
        noun_decl = row[1]
        pronunciation = row[2]
        ipa = row[3]
        sound = row[4]
        translation = row[5]
        pos = row[6]
        deck_id = row[7]
        fields = [lemme, noun_decl, pronunciation, ipa, sound, translation, pos, deck_id]

        for i in range(len(fields)):
            if pd.isna(fields[i]):
                fields[i] = ''

        # tags = row[8] # needs the form: ['tag1', 'tag2', 'french', 'frequency_1_500']
        # if pd.isna(tags):
        #     tags = ''

        # format note
        note = genanki.Note(
            model=custom_french_forvo_model,
            fields=fields,
            # tags=tags
        )

        # add note to deck
        deck.add_note(note)
    return

if __name__ == '__main__':
    main()