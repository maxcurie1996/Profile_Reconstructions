from DataPrep import *
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd

from SULI2021.random_tools.DataPrep import DataPrep


def minmax():
    num_elms = []
    t_inter_elm = []
    for file in os.listdir('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'):
        file = file.split('.')[0]
        sh = DataPrep(file, dir)
        elms = sh.elm_loc()
        if len(elms) < 5:
            continue

        num_elms.append(len(elms))
        t_inter_elm = np.concatenate((t_inter_elm, np.diff(elms)), axis=None)

    print('Average number of ELMs per file: {0} \u00B1 {1}'.format(np.mean(num_elms), np.std(num_elms)))
    print('Max/Min number of ELMs per file: {0}/{1}'.format(np.max(num_elms), np.min(num_elms)))
    print('Average time between ELMs: {0} \u00B1 {1}'.format(np.mean(t_inter_elm), np.std(t_inter_elm)))
    print('Max/Min Time Between ELMs: {0}/{1}'.format(np.max(t_inter_elm), np.min(t_inter_elm)))

    return


def average_modes():
    num_modes = []
    for file in sorted(os.listdir('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/')):
        if '153' in file:
            print(file, ': EJECTED')
            continue
        print(file)
        file = file.split('.')[0]
        sh = DataPrep(file, dir)

        masked = sh.make_mask()
        print('Mask Complete')
        i = 1
        tot = len(masked)
        for slc in masked:
            nm_slc = np.ma.clump_masked(slc)
            slc_filtered = [i for i in nm_slc if i.stop - i.start > 5]
            num_modes.append(len(slc_filtered))
            print('\r{0}: {1} of {2} Completed - {3}%'.format(file, i, tot, 100 * i / tot), end='')
            i += 1
        print('\n')
    print('Average Number of Modes at a Given Time: {0} \u00B1 {1}'.format(np.mean(num_modes), np.std(num_modes)))


def make_interpolation(sh_obj: DataPrep, ielm_df, xdimension, ydimension, plot=False, ax=None):

    if not hasattr(sh_obj, 'mask'):
        sh_obj.make_mask()
        print('hassatr failed')

    ielm_df = ielm_df[1]
    ielm_no = ielm_df.index[0][0]
    ielm_index = np.array([i[2] for i in ielm_df.index])
    ielm_time = np.array([i[1] for i in ielm_df.index])
    ielm_mask = sh_obj.mask[ielm_index][:, :sh_obj.stop_height]
    # print(ielm_df.index[0], ielm_df.index[-1], ielm_no, ielm_df.index[-1][0])
    # print('\rAnalyzing shot {0}/{1} ({2}): {3}% complete'.format(shots.index(shot),
    #                                                              len(shots),
    #                                                              shot,
    #                                                              100 * ielm_no / elm_df.index[-1][0]),
    #       end=' ')

    x = np.linspace(0, xdimension, np.size(ielm_mask, axis=0))
    y = np.linspace(0, ydimension, np.size(ielm_mask, axis=1))

    interp_function = interpolate.interp2d(x, y, ielm_mask.T, kind='linear')

    x_new = np.arange(0, xdimension)
    y_new = np.arange(0, ydimension)

    z_new = interp_function(x_new, y_new)

    if plot:
        if ax is None:
            ax = plt.gca()

        heatmap = ax.imshow(z_new,
                            origin='lower',
                            extent=[ielm_time[0], ielm_time[-1], 0, sh_obj.stop_height],
                            aspect='auto')
        ax.set_title('Shot Number: {0}; ELM Number:{1}'.format(sh_obj.shot_no, ielm_no))
        ax.set_xlabel('Time (ms)')
        ax.text(0.1, 0.9, 'x_dim = {0}\ny_dim = {1}'.format(xdimension, ydimension),
                horizontalalignment='left',
                verticalalignment='top',
                color='white', fontsize=16,
                transform=ax.transAxes)
        plt.show()

    return z_new


def make_histogram(shots, plot=False, ax=None):
    xdim = 2000
    ydim = 2000
    total = np.zeros((xdim, ydim))
    shots = [i.split('.')[0] for i in shots]

    for shot in shots:
        if '153' in shot:
            print(shot + ': REJECTED')
            continue
        else:
            print('\nBeginning analysis shot {0}:'.format(shot), end=' ')

        print('Getting shot data'.format(shot), end=' ')
        sh: DataPrep = DataPrep(shot, dir)
        sh.set_mask_binary = True
        print('\rPreparing analysis shot {0}: Detecting ELMs'.format(shot), end=' ')
        elm_df = sh.split()
        print('\rPreparing analysis shot {0}: Making masked array'.format(shot), end=' ')
        mask = sh.make_mask()
        print('\rPreparing analysis shot {0}: Ready.'.format(shot))

        for ielm_df in elm_df.groupby(level=0):
            z_new = make_interpolation(sh, ielm_df, xdim, ydim)
            total = total + z_new
            
    if plot:
        if ax is None:
            ax = plt.gca()
    
        ax.imshow(total,
                  origin='lower',
                  extent=[0, xdim, 0, sh.stop_height],
                  aspect='auto')
        ax.set_title('Composite Locations of all ELMs Overlaid')
        ax.set_xlabel('Normalized Time to {}'.format(xdim))
        ax.text(0.1, 0.9, 'x_dim = {0}\ny_dim = {1}'.format(xdim, ydim),
                horizontalalignment='left',
                verticalalignment='top',
                color='white', fontsize=16,
                transform=ax.transAxes)
        plt.show()

    return total


def list_widths(window: pd.DataFrame):

    widths = window['Left/Right'].apply(lambda x: list(map(lambda y: y[1]-y[0], x)))

    return widths


if __name__ == '__main__':

    global dir
    dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
    shot = 174830
    sh = DataPrep(shot)
    window_num = 1

    window = sh.peak_properties().xs(window_num, level=0)
    window['Widths'] = window['Left/Right'].apply(lambda x: [*map(lambda y: y[1]-y[0], x)])
    print(window['Widths'].tolist())
    exit()

    shots = sorted(os.listdir('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'))
    make_histogram(shots)
    shots = [i.split('.')[0] for i in shots]

    numelmslst = []
    timelms = []
    for shot in shots:
        if '153' in shot:
            print(shot + ': REJECTED')
            continue

        sh = DataPrep(shot)
        elms = sh.elm_loc()

        elm_diff = np.diff(elms)
        elm_filtered = elm_diff[elm_diff > 50]

        numelmslst.append(len(elm_filtered))
        timelms = np.concatenate((timelms, elm_filtered), axis=None)
        print(shot)
        print('Number of elms: ', len(elm_filtered))
        print('Mean time between elms: ', np.mean(elm_filtered))
        print('Median intraelm time: ', np.median(elm_filtered))

    print('Total Number of ELMS: ', np.sum(numelmslst))
    print('Average number of elms: ', np.mean(numelmslst))
    print('Average Intra-ELM Time: ', np.mean(timelms))
    print('Median Intra-ELM time: ', np.median(timelms))

    # make_histogram(shots)
