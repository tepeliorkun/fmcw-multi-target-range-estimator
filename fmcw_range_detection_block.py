import numpy as np
from gnuradio import gr
from scipy.signal import find_peaks

class blk(gr.sync_block):
    
#radar menzil için parametreler
    def __init__(self, samp_rate=25e6, fft_size=1024, slope=2.5e11):
        gr.sync_block.__init__(
            self,
            name=' Multi-Target FMCW Range Estimator',
            in_sig=[(np.float32, fft_size)],
            out_sig=[(np.float32, 3)]
        )
        self.samp_rate = samp_rate
        self.fft_size = fft_size
        self.slope = slope
        self.c = 3e8
        self.counter = 0

    def work(self, input_items, output_items): 
        mag = input_items[0][0]
        mag_half = mag[:self.fft_size // 2]   # sadece pozitif frekanslar

        # Tepe bul
        peaks, _ = find_peaks(mag_half, height=np.max(mag_half)*0.2, distance=5)

        if len(peaks) == 0:
            for i in range(3):
                output_items[0][i] = 0.0
            return 1

        # Genliklerine göre sırala
        heights = mag_half[peaks]
        sorted_indices = np.argsort(heights)[-3:]
        top_peaks = peaks[sorted_indices]
        top_peaks = np.sort(top_peaks)

        delta_f = self.samp_rate / self.fft_size
        ranges = []
        for peak_bin in top_peaks:
            f_beat = peak_bin * delta_f
            rng = (f_beat * self.c) / (2 * self.slope)
            ranges.append(rng)

        for i in range(3):
            output_items[0][i] = ranges[i] if i < len(ranges) else 0.0

        self.counter += 1
        if self.counter % 20 == 0:
            print("Detected Targets:")
            for i, r in enumerate(ranges):
                print(f"  Target {i+1}: {r:.2f} meters")

        return 1
