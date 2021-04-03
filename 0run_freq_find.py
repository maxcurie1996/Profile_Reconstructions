import numpy as np 			#import numpy library and give it a short name as np
import sys as sys
import matplotlib.pyplot as plt

sys.path.insert(1, './SULI2021/random_tools')
from DataPrep import *
from tools import data_labeler
from tools import flatten_2D
from max_stat_tool import smooth_with_dev

#Take the suggestion from Jeff

#Created by Max Curie 3/25/2021

#python /u/mcurie/Scripts/tokaml-JeffsBranch/0run_freq_find.py
#shot_dir='/p/datad2/gdong/signal_data_ELM/d3d/d3d/B3'
shot_dir='./'
#for current folder use './'
shot_no=170882
marker_size=5	#Size of the dot marker in the plot
bin_size=5
blur_setting=3
overlap_percent_setting=30.

#shot = 170882 # this is the shot number you're looking at
sh = DataPrep(shot_no, shot_dir) # this creates the DataPrep Object
properties = sh.peak_properties(blur=blur_setting)
window_num = 0 # this is the specific window you're looking for. For a list of available window_num, do:
#print(properties)
print(properties.index.get_level_values(level=0))

#From https://stackabuse.com/writing-to-a-file-with-pythons-print-function/
original_stdout = sys.stdout # Save a reference to the original standard output
with open('output.txt', 'w') as f:
	sys.stdout = f # Change the standard output to the file we created.
	#From: https://stackoverflow.com/questions/19124601/pretty-print-an-entire-pandas-series-dataframe
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
		print(properties)
	sys.stdout = original_stdout # Reset the standard output to its original value

window = properties.xs(window_num, level=0) # This retrieves only the data for the specified window_num
window['Widths'] = window['Left/Right'].apply(lambda x: [*map(lambda y: y[1]-y[0], x)]) #This adds a new column to the window dataframe containing a list of all the widths of the modes.

index = np.array(window.index.get_level_values(level=0).tolist())
index = index.astype(float)

percentages = (index-np.min(index))/(np.max(index)-np.min(index))
locations = window['Peak'].tolist()
widths = window['Widths'].tolist()

pre_label_dataset=[]
#pre_label_dataset[time_index]= [list(location), list(width)]

for i in range(len(percentages)):
	pre_label_dataset.append([locations[i],widths[i]])

#print(percentages)




post_label_dataset, total_band_num=data_labeler(pre_label_dataset,overlap_percent=overlap_percent_setting)

#post_label_dataset[time_index]= [list(location), list(width), list(label), list(band_type)]
'''
band_type:  0: constant band from the beginning
            n0x: split, 101: parents label is 1, split once, 
                        203, parents label is 3, split twice
            2: new band
'''
band_labels=[]
#band_labels=[band_label_number,time,info]

#loop through all band numbers
for i in (np.array(range(total_band_num))+1):
	band_temp=[]
	#band_temp[band_num]=[time_index,percent of the ELM, Frequency, width, band_type]
	#loop through all the time steps
	for j in range(len(post_label_dataset)):
		#loop through all the peak of this time step
		for k in range(len(post_label_dataset[j][2])):
			#if post_label_dataset = the current band loop index
			if post_label_dataset[j][2][k]==i:
				band_temp.append([int(index[j]), 				#time step
								percentages[j], 			#percent of the ELM
								post_label_dataset[j][0][k],#Frequency(location of peak) 
								post_label_dataset[j][1][k],#band width(width of peak) 
								post_label_dataset[j][3][k] #band_type
								])
	band_labels.append(np.array(band_temp))

#band_labels=np.array(band_labels)

color_list=['b', 'g', 'r', 'c', 'm', 'y', 'k']


#From https://stackabuse.com/writing-to-a-file-with-pythons-print-function/
original_stdout = sys.stdout # Save a reference to the original standard output
with open('band_labels.txt', 'w') as f:
	sys.stdout = f # Change the standard output to the file we created.
	#From: https://stackoverflow.com/questions/19124601/pretty-print-an-entire-pandas-series-dataframe
	for i in range(total_band_num):
		print('********'+str(i)+'******')
		print('band_temp[band_num]=[time_index,percent of the ELM, Frequency, width, band_type]')
		print('***************')
		print('***************')
		print('***************')
		print('***************')
		print(str(band_labels[i]))
	sys.stdout = original_stdout # Reset the standard output to its original value


print('np.shape(band_labels[0])'+str(np.shape(band_labels[0])))


#pre_label_dataset[time_index]= [list(location), list(width)]
y_location=[]
y_width=[]
for i in pre_label_dataset:
	for j in i[0]:
		y_location.append(j)
	for j in i[1]:
		y_width.append(j)
print('np.shape(y_location)'+str(np.shape(y_location)))

#post_label_dataset[time_index]= [list(location), list(width), list(label), list(band_type)]
y_location=[]
y_width=[]
y_width_avg=[]
y_width_std=[]
for i in post_label_dataset:
	y_width_temp=[]
	for j in i[0]:
		y_location.append(j)
	for j in i[1]:
		y_width.append(j)
		y_width_temp.append(j)
	width_avg=np.mean(y_width_temp)
	width_std=np.std(y_width_temp)
	y_width_avg.append(width_avg)
	y_width_std.append(width_std)

y_width_avg_smooth,y_width_std_smooth=smooth_with_dev(y_width_avg,y_width_std,bin_size)

y_width_avg_plot=[]
y_width_std_plot=[]

for i in range(len(y_width_avg_smooth)):
	if i%bin_size==0:
		y_width_avg_plot.append(y_width_avg_smooth[i])
		y_width_std_plot.append(y_width_std_smooth[i])

print('np.shape(y_location)'+str(np.shape(y_location)))
#total_band_num=10
'''
plt.clf()
plt.plot()
plt.plot(y_location,'o', linestyle='none',alpha=0.4)
#plt.title('Title',fontsize=20)
plt.xlabel('time')
plt.ylabel('frequency')
plt.show()



plt.clf()
plt.plot()
plt.plot(y_width,'o', linestyle='none',alpha=0.4)
#plt.title('Title',fontsize=20)
plt.xlabel('time')
plt.ylabel('width')
plt.show()
'''


plt.clf()
plt.plot()
plt.errorbar(np.arange(0,len(y_width_avg_plot))/float(len(y_width_avg_plot)),y_width_avg_plot,\
	yerr=y_width_std_plot,marker='o',ms=marker_size,linestyle='none')   
#plt.title('Title',fontsize=20)
plt.xlabel('Percent of ELM')
plt.ylabel('width')
plt.show()

#post_label_dataset[time_index]= [list(location), list(width), list(label), list(band_type)]
#band_labels=[band_label_number,time,info]
#band_temp[band_num]=[time_index,percent of the ELM, Frequency, width, band_type]


plt.clf()
plt.plot()
for i in range(total_band_num):
	if len(np.shape(band_labels[i]))==2:
		#plt.errorbar(band_labels[i][:,0],band_labels[i][:,2],\
		#	yerr=band_labels[i][:,3],marker='o',ms=marker_size,linestyle='none')   

		plt.plot(band_labels[i][:,0],band_labels[i][:,2],'o',\
			color=color_list[i%len(color_list)],linestyle='none',\
			alpha=0.4,label='band '+str(i))
#plt.legend()
#plt.title('Title',fontsize=20)
plt.xlabel('time index')
plt.ylabel('frequency')
plt.show()

'''
plt.clf()
plt.plot()
for i in range(total_band_num):
	if len(np.shape(band_labels[i]))==2:
		plt.plot(band_labels[i][:,0],band_labels[i][:,2],'o',\
			color=color_list[i%len(color_list)],linestyle='none',\
			alpha=0.4,label='band '+str(i))
plt.legend()
#plt.title('Title',fontsize=20)
plt.xlabel('time index')
plt.ylabel('frequency')
plt.show()

'''

plt.clf()
plt.plot()
for i in range(total_band_num):
	if len(np.shape(band_labels[i]))==2:
		plt.plot(band_labels[i][:,1]*100.,band_labels[i][:,3],'o',\
			color=color_list[i%len(color_list)],linestyle='none',\
			alpha=0.4,label='band '+str(i))
plt.legend()
#plt.title('Title',fontsize=20)
plt.xlabel('Percent of ELM')
plt.ylabel('Width of frequency band')
plt.show()
