import os
import numpy as np
import seisbench.models as sbm
import torch
from matplotlib import pyplot as plt
from obspy.core import AttribDict
from obspy.signal.filter import bandpass


class StreamData:
    def __init__(self, station, stream=None):
        self.station = station
        self.original_stream = stream
        self.processed_stream = None
        self.annotated_stream = None
        self.picked_signals = None

    def process_detrend_demean(self):
        self.processed_stream.detrend("demean")

    def process_detrend_linear(self):
        self.processed_stream.detrend("linear")

    def process_remove_outliers(self, threshold_factor=2):
        for trace in self.processed_stream:
            # Calculate global mean and standard deviation for the trace
            global_mean = trace.data.mean()
            global_std = trace.data.std()

            # Identify outliers as points beyond the threshold times the standard deviation from the mean
            outliers = np.abs(trace.data - global_mean) > (threshold_factor * global_std)
            trace.data[outliers] = global_mean  # Replace outliers with the global mean

    def process_apply_bandpass(self, freqmin=1, freqmax=40, corners=5):
        for trace in self.processed_stream:
            df = trace.stats.sampling_rate
            trace.data = bandpass(trace.data, freqmin=freqmin, freqmax=freqmax, df=df, corners=corners)

    def process_taper(self, max_percentage=0.05, taper_type="hann"):
        for trace in self.processed_stream:
            trace.taper(max_percentage=max_percentage, type=taper_type)

    def process_denoise(self):
        # Get pretrained model for denoising
        model = sbm.DeepDenoiser.from_pretrained("original")

        # Enable GPU processing if available
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = False
            model.cuda()
            print("CUDA available. Denoising using GPU")
        else:
            print("CUDA not available. Denoising using CPU")

        # Save original channel names
        original_channels = [tr.stats.channel for tr in self.processed_stream]

        # Apply denoising model
        annotations = model.annotate(self.processed_stream)

        # Restore original channel names
        for tr, channel in zip(annotations, original_channels):
            tr.stats.channel = channel

        self.processed_stream = annotations

    def predict_and_annotate(self):
        # Get pretrained model for phase picking
        model = sbm.EQTransformer.from_pretrained("original")

        # Enable GPU processing if available
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = False
            model.cuda()
            print("CUDA available. Phase-picking using GPU")
        else:
            print("CUDA not available. Phase-picking using CPU")

        # Perform classification to extract picks
        outputs = model.classify(self.processed_stream)
        predictions = []

        for pick in outputs.picks:
            pick_dict = pick.__dict__
            pick_data = {
                "peak_time": pick_dict["peak_time"],
                "peak_confidence": pick_dict["peak_value"],
                "phase": pick_dict["phase"]
            }
            predictions.append(pick_data)

        # Perform annotation to visualize the picks within the stream
        annotated_stream = model.annotate(self.processed_stream)

        # Adjust channel names in the annotated stream to include the original channel plus the model suffix
        for tr in annotated_stream:
            parts = tr.stats.channel.split('_')
            if len(parts) > 1:
                tr.stats.channel = '_' + '_'.join(parts[1:])  # Join parts starting from the first underscore

        self.picked_signals = predictions
        self.annotated_stream = annotated_stream

    def filter_confidence(self, p_threshold, s_threshold):
        # Filter detections based on threshold conditions
        self.picked_signals = [
            detection for detection in self.picked_signals
            if (detection['phase'] == "P" and detection['peak_confidence'] >= p_threshold) or
               (detection['phase'] == "S" and detection['peak_confidence'] >= s_threshold)
        ]

    def save_stream_as_miniseed(self, station, stream_to_save, identifier="processed", path=None):
        if not stream_to_save:
            raise ValueError("No stream to save.")

        channel = "*Z*"
        location = "*"

        # Format the filename with NSLC and suffix based on file_type
        nslc = f"{station.network}.{station.code}.{location}.{channel}".replace("*", "")
        filename = f"{station.report_date.strftime('%Y-%m-%d')}_{nslc}.{identifier}.mseed"

        # Determine the target directory
        target_path = path if path else station.date_folder

        filepath = os.path.join(target_path, filename)

        # Ensure the directory exists
        os.makedirs(target_path, exist_ok=True)

        # 确认数据的 dtype
        dtype = stream_to_save[0].data.dtype
        encoding = None

        # 根据数据类型选择合适的编码
        if dtype == 'int32':
            encoding = 'STEIM2'
        elif dtype == 'float32':
            encoding = 'FLOAT32'
        elif dtype == 'float64':
            encoding = 'FLOAT64'
        elif dtype == 'int16':
            encoding = 'STEIM1'
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

        # 设置每个 trace 的编码
        for trace in stream_to_save:
            if not hasattr(trace.stats, 'mseed'):
                trace.stats.mseed = AttribDict()
            trace.stats.mseed.encoding = encoding

        # Write the stream to a MiniSEED file
        stream_to_save.write(filepath, format='MSEED')
        print(f"Stream saved to {filepath}")

    def save_stream_as_png(self, station, stream_to_save, identifier="processed", path=None):
        if not stream_to_save:
            raise ValueError("No stream to save.")

        channel = "*Z*"
        location = "*"

        # Format the filename with NSLC and suffix based on identifier
        nslc = f"{station.network}.{station.code}.{location}.{channel}".replace("*", "")
        filename = f"{station.report_date.strftime('%Y-%m-%d')}_{nslc}.{identifier}.png"

        # Determine the target directory
        target_path = path if path else station.date_folder

        filepath = os.path.join(target_path, filename)

        # Ensure the directory exists
        os.makedirs(target_path, exist_ok=True)

        # Plot the stream and save it as a PNG
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            stream_to_save.plot(ax=ax, show=False)  # Make sure to use the ax parameter
            plt.savefig(filepath, format='png', bbox_inches='tight')
            plt.close(fig)
            print(f"Stream saved as PNG to {filepath}")
        except Exception as e:
            print(f"Failed to save stream as PNG: {e}")

        return filepath
