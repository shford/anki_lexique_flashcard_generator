"""
@Purpose:   Fills anki card 'Sound:' field with downloaded forvo text
            using the most popular result from Forvo.

            This may be preferred to HyperTTS as it makes you
            choose a regional dialect that may miss if the audio
            is only available from a different region.
"""
import json
import time

import requests
from anki.collection import Collection

from s6_Import_Packages_Into_Anki import PROFILE
from s6_Import_Packages_Into_Anki import get_collection_path
from s8_DeNoise_Forvo_Audio import ANKI_DIR as ANKI_MEDIA_DIR


# ==== CONFIGURATION ====
FORVO_API_KEY = 'COPY_PASTE_KEY_HERE'
# ========================


def main():
    forvo_api__standard_pronunciation()


def forvo_api__standard_pronunciation():
    global FORVO_API_KEY

    # initialize collectoin
    col_path = get_collection_path(PROFILE)
    col = Collection(col_path)

    # forvo api information
    api_format = 'json'
    action = 'standard-pronunciation'
    key = FORVO_API_KEY
    lang = 'fr'

    # download audio and update card's 'Sound' field
    text = """événement
            éponge
            statuer
            coagulation
            avatar"""

    words = text.splitlines()
    for word in words:
        time.sleep(0.2)
        word = word.strip()
        if word is None or word == '':
            continue
        url = f'https://apicommercial.forvo.com/key/{key}/format/{api_format}/action/{action}/word/{word}/language/{lang}'
        r = requests.get(url)
        if r.status_code == 200:
            json_data = json.loads(r.text)
            # no pronunciation found
            if len(json_data['items']) == 0:
                print(word)
                continue
            audio_url = json_data['items'][0]['pathmp3']
            audio_country = json_data['items'][0]['country'].lower()
            audio = requests.get(audio_url)
            audio_name = f'{word}_{lang}_{audio_country}.mp3'
            audio_path = f'{ANKI_MEDIA_DIR}/{audio_name}'

            # audio file too small, almost certainly malformed
            if len(audio.content) < 1024:
                print(word)
                continue

            with open(audio_path, 'wb') as f:
                f.write(audio.content)

            # add [audio_path] to 'Sound' field
            search_query = 'Sound:'
            note_ids = col.find_notes(search_query) # 'Return note ids matching the provided search'
            for note_id in note_ids:
                note = col.get_note(note_id)        # 'Get note by note id'
                note.fields[4] = f'[sound:{audio_name}]'
                col.update_note(note)               # 'Save note changes to database'

                # you would think this belongs after the for-loop
                # for efficiency...
                # you would be horribly mistaken.
                # evidently only one note can exist at a time.
                # if the first word were 'envoler' then one could,
                # say, wind up with 8190 cards all with the
                # same audio file, '[sound:envoler_fr_canada.mp3]'
                col.after_note_updates(list(note_ids), True)  # 'If notes modified directly in database, call this afterwards'
        else:
            print(word)

if __name__ == '__main__':
    main()