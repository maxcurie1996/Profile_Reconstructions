# from random_tools.DataAnalysis import *
from random_tools.DataPrep import *

    
if __name__ == '__main__':

    '''
        Shots: 
         [174828.]
         [174829.]
         [174830.]
         [174833.]
         [174860.]
         [174870.]
    '''

    fig, (ax1, ax2) = plt.subplots(2,1)
    sh178430 = DataPrep(174830)


    # exit()

    # mask = sh178430.make_mask()
    elmdf = sh178430.split()
    print(sh178430.time_to_elm())

    mask = sh178430.make_mask()
    elm_loc = sh178430.elm_loc()
    ielm_index = np.array([i[2] for i in elmdf.index])
    ielm_time = np.array([i[1] for i in elmdf.index])

    ax1.imshow(mask[ielm_index].T,
               norm=LogNorm(),
               origin='lower',
               cmap='Set1',
               interpolation='none',
               aspect='auto')
    ax2.imshow(elmdf.to_numpy().T,
               norm=LogNorm(),
               origin='lower',
               cmap='Set1',
               interpolation='none',
               aspect='auto')


    for elm in elm_loc:
        elm_index = np.argmin(np.abs(np.array(ielm_time - elm)))
        ax1.axvline(elm_index, c='r')
        ax2.axvline(elm_index, c='r')

    plt.show()
    # sh178430.plot_slice(tme=1600)
    # elmdf = sh178430.split(plot=True)
    # print(elmdf)

