import json
import os
import subprocess
import time

import av
import numpy


from s8_DeNoise_Forvo_Audio import SELECTED_AUDIO_PREFIX
from s8_DeNoise_Forvo_Audio import ALT_TXT_IN_AUDIO
from s8_DeNoise_Forvo_Audio import WRITE_FORMAT
from s8_DeNoise_Forvo_Audio import USER_PATH
from s8_DeNoise_Forvo_Audio import PROFILE
from s8_DeNoise_Forvo_Audio import LUFS_NORMALIZATION_LEVEL


PYAV_TEST_DIR = f'{USER_PATH}/.local/share/Anki2/{PROFILE}/PYAV_TEST_DIR'


def main():
    t1 = time.time()

    dir_contents = os.listdir(PYAV_TEST_DIR)
    hypertts_mp3_filenames = [f for f in dir_contents if (os.path.isfile(os.path.join(PYAV_TEST_DIR, f)) and SELECTED_AUDIO_PREFIX in f)]
    std_forvo_api_mp3_filenames = [f for f in dir_contents if (os.path.isfile(os.path.join(PYAV_TEST_DIR, f)) and ALT_TXT_IN_AUDIO in f and WRITE_FORMAT in f and not 'ATTS ' in f)]
    mp3_filenames = hypertts_mp3_filenames + std_forvo_api_mp3_filenames

    print(f'Executing denoising program on {len(mp3_filenames)} files.\n')
    # with Pool(processes=cpu_count()) as pool:
    #     pool.map(process_audio_file, mp3_filenames)
    for filename in mp3_filenames:
        process_file(filename)

    t2 = time.time()
    print(f'\nWrote {len(mp3_filenames)} new files in {t2-t1} seconds.\n.')


def measure_loudness_pcm(pcm_bytes, channels, rate, target_lufs=-16, lra=7, tp=-1.5):
    """measure EBU R128 loudness on raw PCM bytes"""
    cmd = [
        'ffmpeg', '-hide_banner', '-nostdin',
        '-f', 's16le', '-ar', str(rate), '-ac', str(channels),
        '-i', 'pipe:0',
        '-af', f'loudnorm=I={target_lufs}:LRA={lra}:TP={tp}:print_format=json',
        '-f', 'null', '-'
    ]

    try:
        result = subprocess.run(cmd, input=pcm_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        pcm_bytes = result.stdout
    except subprocess.CalledProcessError as e:
        print('Stderr:\n')
        print(e.stderr.decode('utf-8'))
        raise

    stderr = result.stderr.decode('utf-8', errors='ignore')
    i = stderr.find('{')
    j = stderr.rfind('}')
    if i == -1 or j == -1:
        raise RuntimeError('could not parse loudnorm json from ffmpeg stderr')
    return json.loads(stderr[i:j+1])


def process_file(filename):
    input_path = f'{PYAV_TEST_DIR}/{filename}'
    with av.open(input_path) as container:
        # 1) decode input mp3 to raw PCM frames
        input_stream = container.streams.audio[0]
        a_rate = input_stream.codec_context.sample_rate
        a_channels = input_stream.codec_context.channels
        a_layout = input_stream.codec_context.layout
        a_format = input_stream.codec_context.format
        pcm_frames = [frame for frame in container.decode(audio=0)]

        # 2) first graph: arnndn -> adeclick
        graph_one = av.filter.Graph()
        rnnn_model_path = '../resources/rnnoise_models/speech_recording.rnnn'

        graph_one.link_nodes(
            graph_one.add_abuffer(template=input_stream),
            add_arnndn_to_graph(graph_one, rnnn_model_path),
            add_adeclick_to_graph(graph_one),
            graph_one.add('abuffersink'),
        ).configure()

        # 3) push frames through graph one
        # for frame in container.decode(input_stream):
        # for frame in pcm_frames:
        #     graph_one.push(frame)
        #     while True:
        #         try:
        #             p_frame = graph_one.pull()
        #             if p_frame is None:
        #                 break
        #             processed_graph_one_frames.append(p_frame)
        #         except:
        #             time.sleep(0.001)
        processed_frames = []
        frame_iter = iter(pcm_frames)
        done_input = False

        def process_frames():
            has_frames_to_push = True
            while True:
                # try to push next input frame, if available
                if has_frames_to_push:
                    try:
                        frame = next(frame_iter)
                        graph_one.push(frame)
                    except StopIteration:
                        has_frames_to_push = False
                        graph_one.push(None)  # signal end of input
                    except:
                        # in case of implementation that's not like fsm and not done
                        # but was actually just blocking
                        continue

                # poll to pull available frames
                while True:
                    try:
                        f = graph_one.pull()
                        if f is None and not has_frames_to_push:
                            # done.
                            return processed_frames
                        elif f is not None:
                            processed_frames.append(f)
                        break # break if not done - to poll or push more frames
                    except av.BlockingIOError:
                        # graph is not ready for more output yet,
                        # attempt to push more frames (good if like fsm),
                        # polls if not able to push
                        break
                    except av.EOFError:
                        # some implementations let you know they're done via this error :')
                        return processed_frames

        processed_frames = process_frames()

        # 4) convert frames to raw PCM and measure loudness
        tgt_layout = 's16'
        pcm_bytes = convert_frames_to_pcm_bytes(processed_frames, tgt_layout, a_layout, a_rate)

        # 5) first-pass loudnorm measurement
        meas = measure_loudness_pcm(pcm_bytes, channels=a_channels, rate=a_rate, target_lufs=LUFS_NORMALIZATION_LEVEL)

        # 6) second graph: loudnorm -> equalizer -> mp3
        output_path = f'{PYAV_TEST_DIR}/PYAV__{filename}'
        with av.open(output_path, mode='w', format='mp3') as out_mp3:
            mp3_stream = out_mp3.add_stream('libmp3lame', rate=a_rate)

            graph_two = av.filter.Graph()

            graph_two.link_nodes(
                graph_two.add_abuffer(template=input_stream),
                add_loudnorm_to_graph(graph_two, meas),
                graph_two.add('abuffersink'),
            ).configure()

            equalizer_node = add_equalizer_to_graph(graph_two),

            # feed the original PCM frames into the second graph
            for frame in processed_frames:
                graph_two.push(frame)
                while True:
                    f = graph_two.pull()
                    if f is None:
                        break
                    for pkt in mp3_stream.encode(f):
                        out_mp3.mux(pkt)

            for pkt in mp3_stream.encode():
                out_mp3.mux(pkt)


def convert_frames_to_pcm_bytes(frames, target_format, target_layout, target_rate):
    pcm_bytes = bytearray()

    for frame in frames:
        # create a resampler to desired format/layout/rate
        resampler = av.audio.resampler.AudioResampler(
            format=target_format,
            layout=target_layout,
            rate=target_rate or frame.sample_rate
        )
        resampled_frames = resampler.resample(frame)  # can be a list

        # extract bytes via numpy
        for resampled in resampled_frames:
            # packed audio stores all channels interleaved in planes[0]
            plane = resampled.planes[0]
            arr = numpy.frombuffer(plane, dtype=numpy.int16)  # match target_format
            pcm_bytes.extend(arr.tobytes())

    return bytes(pcm_bytes)

# -------------------------
# filter graph helpers
# -------------------------
def add_arnndn_to_graph(graph, rel_model_path, mix=1.0):
    abs_model_path = os.path.abspath(rel_model_path)
    return graph.add('arnndn', args=f'model={abs_model_path}:mix={mix}')

def add_adeclick_to_graph(graph):
    adeclick_args = 'w=20:o=90:a=3:t=1.0:b=3:m=a'
    return graph.add('adeclick', args=adeclick_args)

def add_equalizer_to_graph(graph):
    bands = [
        dict(f=120,   w=2,   g=3),
        dict(f=800,   w=1.5, g=2),
        dict(f=3500,  w=2,   g=4),
        dict(f=6000,  w=2,   g=3),
        dict(f=11000, w=1.5, g=2),
    ]
    node = None
    for b in bands:
        eq_node = graph.add('equalizer', args=f'f={b["f"]}:t=q:w={b["w"]}:g={b["g"]}')
        if node is not None:
            node.link_to(eq_node)
        node = eq_node
    return node

def add_loudnorm_to_graph(graph, meas):
    ln_args = (
        f'I={LUFS_NORMALIZATION_LEVEL}:LRA=7:TP=-1.5:'
        f'measured_I={meas["input_i"]}:measured_LRA={meas["input_lra"]}:'
        f'measured_TP={meas["input_tp"]}:measured_thresh={meas["input_thresh"]}:'
        f'offset={meas["target_offset"]}:linear=true:print_format=summary'
    )
    return graph.add('loudnorm', args=ln_args)


if __name__ == '__main__':
    main()