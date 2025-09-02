import av
import io

input_file = "example.mp3"
rnnn_model = "my_model.rnnn"  # path to your tested RNNoise model

# create an in-memory bytes buffer for output
output_buffer = io.BytesIO()

# open input container
with av.open(input_file) as container:
    # create a memory output container in MP3 format
    with av.open(output_buffer, mode='w', format='mp3') as output:
        output_stream = output.add_stream("libmp3lame", rate=44100, channels=2)

        # create filter graph
        graph = av.filter.Graph()

        # buffer source and sink
        src = graph.add_buffer(template=container.streams.audio[0])
        sink = graph.add("abuffersink")

        # chain filters in the correct order: arnndn -> adeclick -> loudnorm -> equalizer
        arnndn = graph.add("arnndn", args=f"model={rnnn_model}")
        adeclick = graph.add("adeclick", args="threshold=0.5")
        loudnorm = graph.add("loudnorm", args="I=-16:LRA=11:TP=-1.5")
        equalizer = graph.add("equalizer", args="f=1000:t=q:w=1:g=3")

        # connect the graph in order
        src.link_to(arnndn)
        arnndn.link_to(adeclick)
        adeclick.link_to(loudnorm)
        loudnorm.link_to(equalizer)
        equalizer.link_to(sink)

        # configure graph
        graph.configure()

        # process frames
        for frame in container.decode(audio=0):
            graph.push(frame)
            while True:
                filtered_frame = graph.pull()
                if filtered_frame is None:
                    break
                for packet in output_stream.encode(filtered_frame):
                    output.mux(packet)

        # flush encoder
        for packet in output_stream.encode():
            output.mux(packet)

# get final MP3 audio as bytes
filtered_mp3_bytes = output_buffer.getvalue()
