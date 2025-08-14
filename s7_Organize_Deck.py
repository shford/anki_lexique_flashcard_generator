from anki.errors import NotFoundError as anki_err_NotFoundError
from anki.collection import Collection

from s6_Import_Packages_Into_Anki import PROFILE
from s6_Import_Packages_Into_Anki import get_collection_path


def main():
    organize_deck()
    return


def organize_deck():
    # initialize collectoin
    col_path = get_collection_path(PROFILE)
    col = Collection(col_path)

    # delete unwanted Placeholder objects from col (deal w/ genanki wankiness)
    model_names = ['Placeholder_Model', 'Placeholder_Model+']
    for model_name in model_names:
        try:
            notetype_id = col.models.id_for_name(model_name)  # returns NotetypeId
            if notetype_id is None:
                continue
            col.models.remove(notetype_id)  # -"Delete model, and all its cards/notes."
        except anki_err_NotFoundError:
            continue


    # move cards to desired subdecks
    model_name = 'Custom French Forvo'  # presumably this is the NoteType
    en_fr_card_name = 'En_Fr'  # (in genanki template name)
    fr_en_card_name = 'Fr_En'  # "
    tgt_card_types = [en_fr_card_name, fr_en_card_name]
    tgt_deck_names = ['English to French', 'French to English']

    # get card IDs for notes with tag x:
    for tgt in range(len(tgt_card_types)):
        card_ids = col.find_cards(f'Card:{tgt_card_types[tgt]}')
        for card_id in card_ids:
            # get card
            card = col.get_card(card_id)

            # find deck of card
            card_deck_id = card.current_deck_id()
            deck = col.decks.get(card_deck_id)
            deck_name = deck['name']

            # skip previously organized cards
            if tgt_deck_names[tgt] in deck_name:
                continue

            # update card's deck
            tgt_deck_name = f'{deck_name}::{tgt_deck_names[tgt]}'
            tgt_deck_id = col.decks.id_for_name(tgt_deck_name)  # get_deck_id_by_name
            col.set_deck([card_id], tgt_deck_id)

    # saving is deprecated but it makes me feel safe inside
    col.save()

    return


if __name__ == '__main__':
    main()