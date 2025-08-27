"""
@Purpose:       Make it easy to run steps 1 through 7.
                s8 will still need to be run after HyperTTS adds audio.
                See README for details.

@Instructions:  Ensure the configurations below are correct. See README for more details.

@Author :       @shford
"""
import os.path

from anki_lexique_flashcard_generator.s1_Filter_Lexique import OUTPUT_CSV
from anki_lexique_flashcard_generator.s1_Filter_Lexique import clean_lexique as s1_clean_lexique
from anki_lexique_flashcard_generator.s3_Export_Anki_Format import export_anki_format_csvs as s3_export_anki_format_csvs
from anki_lexique_flashcard_generator.s4_Enforce_Csv_Num_Rows import enforce_csv_num_rows as s4_enforce_csv_num_rows
from anki_lexique_flashcard_generator.s5_Generate_Anki_Package import generate_anki_packages as s5_generate_anki_packages
from anki_lexique_flashcard_generator.s6_Import_Packages_Into_Anki import import_anki_packages as s6_import_anki_packages
from anki_lexique_flashcard_generator.s7_Organize_Deck import organize_deck as s7_organize_deck

# ==== OVERRIDE CRITICAL CONFIGURATIONS ====
PROFILE = 'User 1'  # set equal to your Anki profile name

"""
Recommend setting to + CHUNKSIZE the amount you actually want.

So if you want 10000 and CHUNK_SIZE = 500, set DESIRED_FLASHCARDS to 10500.
Sometimes the Lexique has invalid data. Having an extra file means those rows
 get shifted up to fill in missing rows during s4. You can always just delete
the unwanted deck from Anki later.

Set this to 0 to create as many flashcards as possible from your Lexique.
"""
DESIRED_FLASHCARDS = 32500

"""
On the first run this should be 1.

Description: For s6 - importing into Anki, makes it so that only package files
              -geq this number are imported. Used when some package files have 
              already been imported and edited and you don't wish to update or
              overwrite those changes.

Example for setting this:
  ; in Easy_Run_s1_s7.py
  START = 1001
  # assume CHUNK_SIZE in s2 is still 500
  
  ; let's assume these are contents of the directories of default_output/anki_lexique_imports
    and default_output/anki_packages from s3-s5
  example_csvs = ['Deck 1 - 500.csv', 'Deck 2 - 500.csv', 'Deck 3 - 500.csv']'
  example_pkgs = ['Deck_1.apkg', 'Deck_2.apkg', 'Deck_3.apkg']
  
  ; in s6_Import_Packages_Into_Anki.py
  # note: start is set to param over global
  packages_to_import = get_packages(start)
  # packages_to_import contents would be ['Deck_2.apkg', 'Deck_3.apkg']
   
"""
START = 1        # see block comment
# ===========================================


def main():
    start_prompt()

    if not os.path.exists(OUTPUT_CSV):
        s1_clean_lexique()

    # no need to run s2 directly, it's called by s3
    s3_export_anki_format_csvs(DESIRED_FLASHCARDS)
    s4_enforce_csv_num_rows()
    s5_generate_anki_packages(DESIRED_FLASHCARDS)
    s6_import_anki_packages(start=START)
    s7_organize_deck()

    print(f'Finished running steps 1-7. {DESIRED_FLASHCARDS} flashcards imported into Anki.')


def start_prompt():
    print('Running easy flashcard generator...')
    print('')
    print('Please verify your profile_name and desired number of flashcards are correct:')
    print(f'PROFILE=\'{PROFILE}\'')
    print(f'DESIRED_FLASHCARDS={DESIRED_FLASHCARDS}')
    print(f'START={START}')
    verification = input('Please enter yes to continue, or anything else to quit: ').lower()

    if not (verification == 'y' or verification == 'yes'):
        print('Exiting...')
        exit(0)


if __name__ == '__main__':
    main()