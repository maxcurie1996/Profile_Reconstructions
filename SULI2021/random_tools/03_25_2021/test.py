import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d



def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def plt_stop_start(elm_slice, split_df, masked):

    single_df = split_df.xs(str(elm_slice), level=0)

    elm_id = {}
    start = single_df.index.get_level_values('Index')[0]
    for i in single_df.index.get_level_values('Index'):
        if i == start:
            list1 = np.ma.flatnotmasked_contiguous(masked[0])
            continue
        else:
            list2 = np.ma.flatnotmasked_contiguous(masked[i])

        for index in range(len(list2)):
            if index not in elm_id:
                elm_id[index] = []
            try:
                if list2[index].start <= list1[index].stop:
                    elm_id[index].append(list2[index])
                elif list2[index].stop >= list1[index].start:
                    elm_id[index].append(list2[index])
            except:
                pass

        list1 = list2

    fig, ax = plt.subplots(1,1)
    for i, x in enumerate(elm_id[1]):
        ax.scatter(np.full(2, i), [x.start, x.stop])

    plt.show()


if __name__ == '__main__':
    arr = np.random.normal(0.003, 0.005, (8192, 10201))
    arr = np.transpose(arr)

    x_vals = np.arange(0, len(arr[0]))
    sigma = 50
    for x in range(len(arr[:,0])):
        arr[x] = arr[x]+gaussian(x_vals, 8000, sigma)/20+\
                 gaussian(x_vals, 6000, sigma)/50+\
                 gaussian(x_vals, 4000, sigma)/100+\
                 gaussian(x_vals, 2000, sigma)/150+\
                 gaussian(x_vals, 500, sigma)/250


    arr_complete = np.array([np.linspace(906, 5084, 10201), np.transpose(arr), np.linspace(0, 1000, 8192)], dtype=object)
    # plot_mask(arr_complete)
    # plot(arr_complete, 2085)


    pindex, pmap = sweep(arr_complete)
    # print(np.shape(pmap))
    sigma = 2*sigma
    print('False positives: ', 100*(np.count_nonzero(pmap)-\
                                (np.count_nonzero(pmap.T[500-sigma:500+sigma])+\
                                np.count_nonzero(pmap.T[2000-sigma:2000+sigma])+\
                                np.count_nonzero(pmap.T[4000-sigma:4000+sigma])+\
                                np.count_nonzero(pmap.T[6000-sigma:6000+sigma])+\
                                np.count_nonzero(pmap.T[8000-sigma:8000+sigma])))/np.count_nonzero(pmap), '%')

    print('% Accuracy lowest amplitude ({}) :'.format(1/250), 100*np.count_nonzero(pmap.T[500])/np.size(pmap.T[500]), '%')
