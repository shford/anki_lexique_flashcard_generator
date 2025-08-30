# Lexique Sound Flashcard Generator For Anki
> Generates high quality French flashcards for Anki.
> 
## Getting Started

If you are not familiar with Anki, you should start here:
<br>
[Get latest release](https://github.com/ankitects/anki/releases)
<br>
[Register with AnkiWeb](https://ankiweb.net/decks)

This project comes bundled with the Lexique and DELA dictionary. However their official projects (with potentially more up-to-date sources) can be found at:
<br>
[Lexique](http://www.lexique.org/)
<br>
[DELA dictionary](https://infolingu.univ-mlv.fr/DonneesLinguistiques/Dictionnaires/telechargement.html)

## Project Description
This project generates French flashcards for Anki.

The language to translate from French to can be configured in s3. It defaults to American English.

The DeepL translation service was chosen for its ease of signup and is free up to a pretty high usage limit. See their website for details.  AWS Translate support may also be coming in future updates.

### Project Structure

Below are the steps this program executes (in files s1..s8).

1. Filter bogus data from the Lexique.
2. Intermix spoken/written frequency.
3. Generate the files that will be imported into Anki.
4. Even out the number of rows in our exported csv's so that each will make exactly 500 (by default) cards.
5. Create packages flashcards using `genanki`.
6. Import package files into Anki.
7. Organize decks.
<br>
(Manually add audio.)
<br>
8. Clean audio created by HyperTTS.

# Installation / Instructions

1. Clone or download this repository.
2. Register for DeepL API (free version is fine).
3. Copy your API key into `deepl_credentials.txt`
4. Register for desired audio API (recommend [Forvo API](https://api.forvo.com])).
5. Config profile name and desired numbers of flashcards in Easy_Run_s1_s7.py.
6. Ensure Anki is closed.
7. Run Easy_Run_s1_s7.py.
8. [Follow guide to add audio from HyperTTS](https://www.youtube.com/watch?v=QYK2GPxksq4)
   - Note: You can use the Forvo API directly as shown below.
     - <i>Hint: Select the 'api:url' field to toggle the url as desired.</i> 
        !['HyperTTS Picture'](hypertts_config.png)

9. Ensure Anki is still closed and no Anki media audio files are open in other applications.
10. Run s8_DeNoise_Forvo_Audio.py to remove background noise and normalize volume.
    - Recommended if you chose an audio service that uses real recordings (i.e. Forvo).

## Contributing

Bug fixes, issue reporting, and PRs are welcome. 

If you're really eager to help, s8 wraps ffmpeg and ffmpeg-normalize calls with subprocess. These commands are shell commands which are natually platform dependant and have a lot of overhead. All that could be replaced with the ffmpeg library with a little Python function binding.


## Licensing/Resources

This project utilizes the Lexique and DELA dictionary. Aside from the inclusion of these projects for convenience, it is not a derivative project of either work, although its output as is will generate derivative works.

<sub>1. Any flashcards generated using the Lexique fall under its derivative license, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This project complies with the Lexique Project's license, makes changes to the Lexique in a reasonable manner, and acknowledges that "in no way does this use suggest that the licensor endorses this project's creator(s) or their use."</sub>

<sub>2. This project packages the original inflected form of the DELA French dictionary provided by the former Laboratoire d'Automatique Documentaire et Linguistique (LADL), now integrated into Institut Gaspard Monge (IGM) of the Université Gustave Eiffel which uses the [LGPLLR]((https://infolingu.univ-mlv.fr/DonneesLinguistiques/Lexiques-Grammaires/lgpllr.html)) license.</sub>

###### Lexique Restriction on Commercial Use of Derivative Works
> "The Lexique and its derivative works "may not [be used]... for commercial purposes."

## Meta
Distributed under the MIT license. Derivative licenses linked as required.

## Disclaimer
This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
