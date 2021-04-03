from numpy import argmin, abs, array, ndarray
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import SULI2021.random_tools.DataPrep as dp
import numpy as np
import copy

def index_match(arr1: ndarray, time: float):
    return argmin(abs(array(arr1) - time))


def plot_t_to_elm(sh_obj):
    elmdf = sh_obj.split()
    print(elmdf.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.imshow(elmdf.to_numpy().T,
               norm=LogNorm(),
               origin='lower',
               interpolation='none',
               aspect='auto')
    ax2.plot(range(len(elmdf.index)), [i[3] for i in elmdf.index])
    plt.show()


def plot_split(sh_obj):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    fig.subplots_adjust(hspace=0)
    fig.suptitle(f'Filtered and Masked Spectrogram for Shot {sh_obj.shot_no}', fontsize=20)

    elmdf = sh_obj.split()
    sh_obj.set_mask_binary = True
    mask = sh_obj.make_mask()
    elm_loc = sh_obj.elm_loc()
    ielm_index = array([i[1] for i in elmdf.index])
    ielm_time = array([i[2] for i in elmdf.index])
    ax2.imshow(mask[ielm_index].T,
               vmin=0, vmax=1,
               origin='lower',
               cmap='Reds',
               alpha=1,
               interpolation='none',
               aspect='auto')
    ax1.imshow(elmdf.to_numpy().T,
               norm=LogNorm(),
               origin='lower',
               interpolation='none',
               aspect='auto')

    ax1.text(0.5, 0.9, 'Spectrogram with ELMs Demarcated',
             horizontalalignment='center',
             verticalalignment='top',
             color='black', fontsize=16,
             transform=ax1.transAxes)
    ax2.text(0.5, 0.9, 'Masked Array with Modes Only',
             horizontalalignment='center',
             verticalalignment='top',
             color='black', fontsize=16,
             transform=ax2.transAxes)

    for i, elm in enumerate(elm_loc[:-1]):
        if elm_loc[i + 1] - elm <= 50:
            continue
        elm_index = argmin(abs(array(ielm_time - elm)))
        ax1.axvline(elm_index, c='r', alpha=0.1)
        ax2.axvline(elm_index, c='r', alpha=0.1)

    plt.show()

def flatten_2D(a):
    b=[]
    for i in a:
        for j in i:
            b.append(j)

    b=np.array(b)
    return b

def data_labeler(a,overlap_percent=50.):
    #a[time_index]= [list(location), list(width)]
    total_label=0
    b=[]
    #loop through the time steps
    for i in range(len(a)):
        print("at timestep "+str(i))
        #for the first time step
        if i==0:
            location_temp=[]
            width_temp=[]
            label_temp=[]
            band_type_temp=[]

            total_label=len(a[i])
            for j in range(total_label):
                location_temp.append(a[i][j][0])
                width_temp.append(a[i][j][1])
                label_temp.append(j)
                band_type_temp.append(0)
        #for the rest of time steps
        else:
            location_temp=[]
            width_temp=[]
            print('np.shape(a[i])'+str(np.shape(a[i])))

            #A new frequency band appears will be labeled -1
            label_temp=[-1]*len(a[i][0])
            band_type_temp=[0]*len(a[i][0])
            #loop throught all the bands for this time step
            for j in range(len(a[i][0])):
                location_temp.append(a[i][0][j])
                width_temp.append(a[i][1][j])

                current_location=a[i][0][j]
                current_width   =a[i][1][j]

                #find the label
                #loop throught all the bands from the last time step
                for m in range(len(b[-1][2])):
                    last_location   =b[-1][0][m]
                    last_width      =b[-1][1][m]
                    last_label      =b[-1][2][m]

                    overlap_percent_calc=100.*abs(current_location-last_location)/\
                                    np.mean([current_width,last_width])
                    
                    print('current_width'+str(current_width))
                    print('last_width'+str(last_width))
                    print('current_location'+str(current_location))
                    print('last_location'+str(last_location))
                    
                    
                    print('overlap_percent_calc='+str(overlap_percent_calc))
                    #Current frequency band
                    if (overlap_percent_calc<=overlap_percent):
                        label_temp[j]=last_label
            #end of loop throught all the bands for this time step

            #start of labeling all the band type 2 and n0x 
            label_count=np.zeros(int(np.max(label_temp))+2)

            #print('label_temp'+str(label_temp))

            for j in label_temp:
                if j==-1:
                    #total number of new bands
                    label_count[0]=label_count[0]+1
                else:
                    label_count[j]=label_count[j]+1

            #print('label_count: '+str(label_count))
            #counter for the loop
            label_counter=np.zeros(int(np.max(label_temp))+int(label_count[0])+1)

            #print('label_counter'+str(label_counter))

            label_temp_copy=copy.deepcopy(label_temp)

            #loop through the label_list from current time step
            for j in range(len(label_temp)):
                #print('label_temp_copy'+str(label_temp_copy))
                #constant band
                if label_count[label_temp[j]]==1:
                    band_type_temp[j]=0
                #Current frequency band split
                     #Multiple bands              and   not new bands
                elif label_count[label_temp[j]]>1 and label_temp[j]!=-1:
                    band_type_temp[j]=100+label_temp[j]
                    k=label_counter[label_temp[j]]
                    #the first branch of the 
                    if k==0:
                        label_temp[j]=label_temp[j]
                    else:
                        total_label=total_label+1
                        label_temp[j]=total_label
                    #print('label_temp_copy'+str(label_temp_copy))
                    #print('label_temp_copy[j]'+str(label_temp_copy[j]))
                    #print('j:'+str(j))
                    label_counter[label_temp_copy[j]]=label_counter[label_temp_copy[j]]+1
                    
                #A new independent frequency band appears
                if label_temp[j]==-1:
                    total_label=total_label+1
                    label_temp[j]=total_label
                    band_type_temp[j]=2

            #print('label_count:'+str(label_count))
            #print('label_counter:'+str(label_counter))
            #end of labeling all the band type 2 and n0x

        b.append([location_temp, width_temp, label_temp, band_type_temp])

    #b[time_index]= [list(location), list(width), list(label), list(band_type)]
    '''
    band_type:  0: constant band from the beginning
                n0x: split, 101: parents label is 1, split once, 
                            203, parents label is 3, split twice
                2: new band
    '''
    return b, total_label