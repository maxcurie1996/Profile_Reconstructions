from numpy import argmin, abs, array
import numpy as np
import copy

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
                #find the label
                #loop throught all the bands from the last time step
                for m in range(len(b[-1][2])):
                    current_location=a[i][0][j]
                    current_width   =a[i][1][j]
                    last_location   =b[-1][0][m]
                    last_width      =b[-1][1][m]
                    last_label      =b[-1][2][m]

                    overlap_percent_calc=100.*abs(current_location-last_location)/\
                                    np.mean([current_width,last_width])
                    #Current frequency band
                    if (overlap_percent<=overlap_percent_calc):
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