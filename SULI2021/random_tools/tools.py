from numpy import argmin, abs, array
import numpy as np

#Modified by Max Curie 03/24/2021

def index_match(arr1, time):
    return argmin(abs(array(arr1) - time))


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
    for i in range(len(a)):
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
        else:
            location_temp=[]
            width_temp=[]
            label_temp=[]
            band_type_temp=[]
            #loop throught all the bands for this time step
            for j in range(len(a[i])):
                location_temp.append(a[i][j][0])
                width_temp.append(a[i][j][1])
                #find the label
                #loop throught all the bands from the last time step
                for m in range(len(b[-1][2])):
                    current_location=a[i][j][0]
                    current_width   =a[i][j][1]
                    last_label      =b[-1][2][m]
                    last_location   =b[-1][2][m]
                    last_width      =b[-1][1][m]

                    overlap_percent_calc=100.*abs(current_location-last_location)/\
                                    np.mean([current_width,last_width])
                    #Current frequency band
                    if (overlap_percent<=overlap_percent_calc):
                        label_temp.append(last_label)
                    #A new frequency band appears
                    else: 
                        label_temp.append(-1)

                label_count=np.zeros(len(b[-1][2]))

                band_type_temp=np.zero(np.shape(label_temp))

                for j in label_temp:
                    label_count[j]=label_count[j]+1

                label_counter=np.zeros(np.shape(label_count))

                #loop through the label_list from current time step
                for j in range(len(label_temp)):
                    #constant band
                    if label_count[label_temp[j]]==1:
                        band_type_temp[j]=0
                    #Current frequency band split
                    elif label_count[label_temp[j]]>1 and label_temp[j]!=-1:
                        band_type_temp[j]=100+label_temp[j]
                        k=label_counter[label_temp[j]]
                        #the first branch of the 
                        if k==0:
                            label_temp[j]=label_temp[j]
                        else:
                            total_label=total_label+1
                            label_temp[j]=total_label
                        label_counter[label_temp[j]]=label_counter[label_temp[j]]+1
                    #A new independent frequency band appears
                    if label_temp[j]==-1:
                        total_label=total_label+1
                        label_temp[j]=total_label
                        band_type_temp[j]=2

                print('label_count:'+str(label_count))
                print('label_counter:'+str(label_counter))

        b.append([location_temp, width_temp, label_temp, band_type_temp])

    #b[time_index]= [list(location), list(width), list(label), list(band_type)]
    '''
    band_type:  0: constant band from the beginning
                n0x: split, 101: parents label is 1, split once, 
                            203, parents label is 3, split twice
                2: new band
    '''
    return b