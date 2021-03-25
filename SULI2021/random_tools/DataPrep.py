import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks, argrelmax, argrelmin, spectrogram
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os
try:
    from tools import index_match
except:
    from .tools import index_match
import pyarrow


class DataPrep:

    def __init__(self, shot_no):

        self.shot_no = shot_no

        # variables that can be changed to tweak data output
        # Minimum time between ELMs
        self.min_elm_window = 50
        # spectrogram parameters
        self.spectrogram_height = 8192
        self.spectrogram_width = 10201
        # Whether to return binary masked array or encode amplitude into mask
        self.set_mask_binary = False
        # height of highest mode
        self.stop_height = 1500

        self.arr = self.get_shot()

    # extract values for specific shot from above list of shots
    # this is for the HDF5 files my mentor gave me, I'll switch it up for the parquet files
    def get_shot_from_mat(self):
        cwd = os.getcwd()
        file = cwd + '/data/RESULTS_ECH_EFFECTS_SPECTRA_B3.mat'
        # import data from .mat file into python numpy arrays
        mat = h5py.File(file, 'r')
        dat_ech = mat['DAT_ECH']
        shlst = mat['shn'][:]

        shindex = [np.where(shlst == s) for s in shlst if self.shot_no in s][0][0][0]

        tds = dat_ech.get('TIME')[shindex][0]
        sds = dat_ech.get('SPECTRUM')[shindex][0]
        fds = dat_ech.get('FREQ')[shindex][0]

        time = mat[tds][0][:]
        spectrum = mat[sds][:]
        freq = mat[fds][0][:]

        return np.array([time, spectrum, freq], dtype=object)

    # This is the one we can use for the parquet files.
    def get_shot(self):
        # change this to where your parquet files are stored
        pq_dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
        file = pq_dir + str(self.shot_no) + '.pq'

        raw = pd.read_parquet(file, engine='pyarrow')
        raw_values = raw['amplitude'].values

        y_height = self.spectrogram_height  # Default = 8192. IMPORTANT: A power of 2 is most efficient

        freq, time, spectrum, = spectrogram(raw['amplitude'].values,
                                            nperseg=y_height * 2,
                                            noverlap=(y_height * 2) - int(
                                                np.floor(((len(raw_values)) / self.spectrogram_width))) + 1
                                            )

        time = 1000 * (raw['time'].values[0] + (raw['time'].values[-1] - raw['time'].values[0]) * (time - time[0]) / (
                time[-1] - time[0]))
        # spectrum = spectrum/np.linalg.norm(spectrum)

        return np.array([time, spectrum, freq], dtype=object)

    def elm_loc(self, plot=False):
        pq_dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
        file = pq_dir + str(self.shot_no) + '.pq'

        raw = pd.read_parquet(file, engine='pyarrow')
        raw_values = raw['amplitude'].values
        raw_time = raw['time'].values
        peaks, props = find_peaks(np.absolute(raw_values), prominence=(None, None), distance=1000,
                                  height=(np.median(abs(raw_values)) * 20, None),
                                  width=(None, None), rel_height=1.0)
        peaks_time = 1000 * raw_time[peaks]

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            fig.subplots_adjust(hspace=0)

            self.heatmap2d(ax=ax1)
            ax2.plot(peaks_time, props['peak_heights'], 'ro')
            ax2.plot(1000*raw_time, raw_values)
            ax1.set_title(None)
            ax1.text(0.5, 0.9, 'Spectrogram',
                    horizontalalignment='center',
                    verticalalignment='top',
                    color='white', fontsize=16,
                    transform=ax1.transAxes)
            ax2.text(0.5, 0.9, 'Raw B-Dot Data',
                    horizontalalignment='center',
                    verticalalignment='top',
                    color='black', fontsize=16,
                    transform=ax2.transAxes)
            ax2.set_xlabel('Time (ms)')
            fig.suptitle(f'Locations of ELMs in shot {self.shot_no}', fontsize=20)

            plt.show()

        return peaks_time

    # make the heatmap
    def heatmap2d(self, ax=None):
        '''
        This basically just plots the spectrogram.
        '''
        if ax is None:
            ax = plt.gca()

        heatmap = ax.imshow(self.arr[1],
                            norm=LogNorm(),
                            origin='lower',
                            extent=[self.arr[0][0], self.arr[0][-1], 0, len(self.arr[2])],
                            interpolation='none',
                            aspect='auto'
                            )
        ax.set_title('Spectrogram')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency')
        return heatmap

    # make the plot of the spectral data for given time
    def slice1d(self, time, smooth=False, ax=None):
        '''
        This plots the 1D slice in plot slice. Not super necessary as an extra function, but I kinda wanted to see if
        I could move matplotlib objects between functions.
        '''
        if ax is None:
            ax = plt.gca()
        index = index_match(self.arr[0], time)  # finds index of time entered. Kind of slow?
        if len(self.arr[0]) < index < 0:
            raise ValueError('Selected time is out of bounds of run time for shot.')
        slce = self.arr[1][:, index]
        ax.plot(self.arr[2], slce)
        ax.set_xlabel('Frequency (kHz)')
        if smooth:
            smooth_arr = gaussian_filter1d(slce, 10)
            peaks, _ = find_peaks(smooth_arr, prominence=(np.mean(abs(smooth_arr)), None), distance=75)
            ax.plot(self.arr[2], smooth_arr, 'y')
            ax.plot(self.arr[2][peaks], smooth_arr[peaks], 'ro')
        else:
            peaks, _ = find_peaks(slce, prominence=(np.mean(abs(slce)), None), distance=75)
            ax.plot(self.arr[2][peaks], slce[peaks], 'ro')
        return peaks

    # helper function
    def peakomatic(self, slce):
        # These properties can be tweaked to fine-tune the results.
        smooth_arr = gaussian_filter1d(slce, 10)
        peaks, properties = find_peaks(smooth_arr, prominence=(np.mean(abs(smooth_arr)), None), distance=75,
                                       height=0,
                                       width=0)

        return peaks, properties['peak_heights'], zip(properties['left_ips'], properties['right_ips']), properties['width_heights']

    # Function to find peaks through spectrogram
    def get_peaks(self):
        '''
        finds and labels the peaks of each 1D slice of spectrogram. These are the locations of the modes.
        '''

        if hasattr(self, 'peaks_index'):
            return self.peaks_index, self.peaks_map

        self.peaks_map = np.zeros_like(self.arr[1].T)
        peaks_index = []
        for i in range(len(self.arr[1][0])):
            slce = self.arr[1][:, i]

            peaks, peak_heights, width, width_heights = self.peakomatic(slce)

            peaks_index.append(peaks)
            for e, w in enumerate(width):
                self.peaks_map[i][round(w[0]):round(w[1])] = width_heights[e]
            self.peaks_map[i][peaks] = peak_heights
        self.peaks_index = np.asarray(peaks_index, dtype=object)
        return self.peaks_index, self.peaks_map

    def plot_slice(self, tme):
        '''
        I made a few different plotting functions. This one plots the spectrogram and a cross section. For testing
        smoothing functions.
        '''
        fig, (ax1, ax2) = plt.subplots(2, 1)
        self.heatmap2d(ax=ax1)
        ax1.axvline(tme, c='r')
        # smooth bool switches between gaussian filtered 1d slice or raw data.
        peaks = self.slice1d(tme, smooth=True, ax=ax2)
        ax1.plot(np.full_like(peaks, tme), np.arange(len(self.arr[2]))[peaks], 'rx')
        plt.show()

    def make_mask(self, plot=False):
        '''
        returns a np.ma.masked object that is an array of equal shape to the original array.
        This array is masked for all data points not belonging to the modes (the horizontal squigglies)
        Currently, it sets all points belonging to modes to 1, but their real values can be used.
        '''

        if hasattr(self, 'mask'):
            return self.mask

        peaks_index, peaks_map = self.get_peaks()
        # creates array of the amplitude values of modes and null values everywhere else
        # to return only location of modes, set self.set_mask_binary to True
        if self.set_mask_binary:
            mask = np.ma.masked_where(peaks_map != 0, peaks_map)
            mask = mask.filled(fill_value=1)
            mask = np.ma.masked_where(mask == 0, mask)
        else:
            mask = np.ma.masked_where(peaks_map == 0, peaks_map)

        mask = np.transpose(mask)

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            self.heatmap2d(ax=ax1)
            ax1.imshow(mask,
                       extent=[self.arr[0][0], self.arr[0][-1], 0, len(self.arr[2])],
                       norm=LogNorm(),
                       origin='lower',
                       cmap='Set1',
                       interpolation='none',
                       aspect='auto')
            ax1.set_title('Spectrogram With Modes Overlaid (1D Gaussian Filter Applied)')
            ax2.imshow(mask,
                       extent=[self.arr[0][0], self.arr[0][-1], 0, len(self.arr[2])],
                       norm=LogNorm(),
                       origin='lower',
                       cmap='Set1',
                       interpolation='none',
                       aspect='auto')
            ax2.set_title('Modes Only (1D Gaussian Filter Applied)')
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Frequency (kHz) mask plot')
            plt.show()

        self.mask = mask.T
        return self.mask


    # returns dict with all elm cycles of a particular shot. Needs work.
    def split_from_spec(self, plot=False):
        '''
        split returns a dataframe of the original array which excludes all arrays belonging to ELM's.
        It also returns a hot array with 0 for all intra-elm indices and 1 for all elm indices (the ones that were excluded
        from the dataframe.
        '''

        _, pm = self.get_peaks()
        stop = self.stop_height  # default 1500
        pm_norm = (pm[:, stop:] - np.amin(pm[:, stop:])) / (np.amax(pm[:, stop:]) - np.amin(pm[:, stop:]))
        sums = np.sum(pm_norm, axis=1)

        # peaks, props = find_peaks(sums, distance=100, prominence=(1, None), width=(None, None), rel_height=1.0)
        # l_elms = props['left_ips']
        # r_elms = props['right_ips']

        elms = self.elm_loc()
        for i, elm in enumerate(elms):
            i_elm = index_match(self.arr[0], elm)
            # l_elms[i] = i_l_elm
            l_elm = np.argmax(np.gradient(sums[i_elm - 50:i_elm + 50])) + i_elm - 50
            r_elm = np.argmin(np.gradient(sums[i_elm - 50:i_elm + 50])) + i_elm - 50
            l_elms[i] = l_elm
            r_elms[i] = r_elm
        #
        # for i, r_elm in enumerate(r_elms):
        #     i_r_elm = index_match(self.arr[0], r_elm)
        #     r_elms[i] = i_r_elm
        # hot = np.zeros_like(self.arr[0])
        # for i in np.column_stack((l_elms, r_elms)):
        #     hot[i[0]:i[1]] = 1
        r_elms = r_elms.astype(int)
        l_elms = l_elms.astype(int)
        if plot:
            fig, ax = plt.subplots(1, 1)
            self.heatmap2d(ax=ax)
            # for i, num in enumerate(hot):
            #     if num == 1:
            #         ax.axvline(self.arr[0][i], c='orange')
            for i in l_elms:
                ax.axvline(self.arr[0][i], ymin=stop / self.arr[1].shape[0], c='red')
            for i in r_elms:
                ax.axvline(self.arr[0][i], ymin=stop / self.arr[1].shape[0], c='green')
            # for elm in elms:
            #     i_elm = index_match(self.arr[0], elm)
            #     ax.axvline(self.arr[0][i_elm], c='blue')
            plt.show()
        # make dict with keys, values, times
        elm_cycles = {}
        for i in range(len(r_elms) - 1):
            for j in self.arr[0][r_elms[i]:l_elms[i + 1]]:
                k = np.argwhere(self.arr[0] == j)[0][0]
                elm_cycles[(i, j, k)] = self.arr[1].T[np.argwhere(self.arr[0] == j)][0][0]

        index = pd.MultiIndex.from_tuples(elm_cycles.keys(), names=['ELM_No', 'Time(ms)', 'Index'])
        elmdf = pd.DataFrame(elm_cycles.values(), index=index)

        return elmdf

    def split(self):

        if hasattr(self, 'elmdf'):
            return self.elmdf

        self.elms = self.elm_loc()
        elm_cycles = {}
        for elm_no, elm_time in enumerate(self.elms[:-1]):

            if self.elms[elm_no + 1] - self.elms[elm_no] <= self.min_elm_window:  # default min_elm_window = 50
                continue

            start_ielm = index_match(self.arr[0], elm_time)
            stop_ielm = index_match(self.arr[0], self.elms[elm_no + 1])

            for ielm_time in self.arr[0][start_ielm:stop_ielm]:
                '''MAX: Uncomment the lines below to get % of ELM'''
                ielm_index = np.argwhere(self.arr[0] == ielm_time)[0][0]
                elm_cycles[(elm_no, ielm_index, ielm_time, self.arr[0][stop_ielm] - ielm_time)] = self.arr[1].T[ielm_index]
                # elm_cycles[(elm_no, ielm_index, ielm_time, (ielm_time-self.arr[0][start_ielm])/(self.arr[0][stop_ielm]-self.arr[0][start_ielm]))] = self.arr[1].T[ielm_index]
        index = pd.MultiIndex.from_tuples(elm_cycles.keys(), names=['ELM_No', 'Index', 'Time (ms)', 'T - ELM (ms)'])
        # index = pd.MultiIndex.from_tuples(elm_cycles.keys(), names=['ELM_No', 'Index', 'Time (ms)', '% ELM'])
        self.elmdf = pd.DataFrame(elm_cycles.values(), index=index)

        return self.elmdf

    def peak_properties(self):

        if not hasattr(self, 'elmdf'):
            self.split()

        props = self.elmdf.apply(lambda x: pd.Series(self.peakomatic(x), index=['Peak', 'Peak Amplitude', 'Left/Right', 'Width Height']), axis=1)

        return props


if __name__ == '__main__':

    sh = DataPrep(174828)

    # print(sh.get_peaks())

    shot = sh.peak_properties().xs(0, level=0)

    percentage = shot.index[3] # (0, 901, 1273.45, 0)
    # Left/Right [(left_index1, right_index1), (left_index2, right_index2)]
    width = shot.loc['Left/Right']