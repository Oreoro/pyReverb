import argparse
import numpy as np
import librosa
from pydub import AudioSegment
from gooey import Gooey, GooeyParser


class AudioProcessor:
    @staticmethod
    def slow_down_audio(input_file, speed_factor):
        sound = AudioSegment.from_file(input_file)

        # Slow down the audio: when speed_factor < 1, speed decreases
        slowed_audio = sound.speedup(playback_speed=1.0/speed_factor)

        # Convert slowed audio to stereo if it's not
        if slowed_audio.channels == 1:
            slowed_audio = slowed_audio.set_channels(2)

        # Convert slowed audio segment to numpy array
        arr = np.array(slowed_audio.get_array_of_samples())

        return arr, slowed_audio.frame_rate

    @staticmethod
    def add_conv_reverb(signal, sr, ir_file):
        ir_signal, ir_sr = librosa.load(ir_file)

        # If either signal is stereo, convert to mono.
        if len(signal.shape) > 1 or len(ir_signal.shape) > 1:
            signal = librosa.to_mono(signal)
            ir_signal = librosa.to_mono(ir_signal)

        # Resample the IR signal to match the sample rate of the input signal
        if ir_sr != sr:
            ir_signal = librosa.resample(ir_signal, ir_sr, sr)

        # Perform convolution
        reverbed_signal = np.convolve(signal, ir_signal)

        return reverbed_signal

    @staticmethod
    def save_to_file(signal, sr, outfile):
        signal_normalized = 0.9 * signal / np.max(np.abs(signal))
        signal_normalized = (signal_normalized * 32767).astype(np.int16)

        audio_segment = AudioSegment(
            signal_normalized.tobytes(), frame_rate=sr, sample_width=2, channels=1)
        audio_segment.export(outfile, format='wav')


@Gooey(program_name="pyReverb by Bilal Arshad Rana",
       default_size=(800, 600),
       navigation='TABBED',
       menu=[{'name': 'About', 'items': [{'type': 'AboutDialog', 'menuTitle': 'About', 'name': 'pyReverb', 'description': 'Add Reverb and slow down Audio Files', 'version': '1.0'}]}])
def main():
    parser = GooeyParser(description='Add Reverb and slow down your Audio File')
    parser.add_argument('Input file', widget="FileChooser",
                        help='Choose the input audio file')
    parser.add_argument('Output file', widget="FileSaver",
                        help='Choose where to save the output audio file')
    parser.add_argument('IR file', widget="FileChooser",
                        help='Choose the Impulse Response audio file.You can download IR files from https://www.openairlib.net/auralizationdb or search google for "free impulse response files"')

    parser.add_argument(
        '--speed', type=np.float64, default=1.0, help="Speed Factor. Lower values slow down the audio.")

    args = parser.parse_args()

    # Slow down audio
    slowed_signal, sr = AudioProcessor.slow_down_audio(args.infile, args.speed)

    # Add convolutional reverb
    reverbed_signal = AudioProcessor.add_conv_reverb(
        slowed_signal.astype(np.float64), sr, args.irfile)

    # Save the output
    AudioProcessor.save_to_file(reverbed_signal, sr, args.outfile)


if __name__ == "__main__":
    main()