import numpy as np 			#import numpy library and give it a short name as np
import sys as sys
sys.path.insert(1, '/u/mcurie/Scripts/tokaml-JeffsBranch/SULI2021/random_tools')
import DataPrep
from tools import data_labeler


#python /u/mcurie/Scripts/tokaml-JeffsBranch/0run_freq_find.py
#shot_dir='/p/datad2/gdong/signal_data_ELM/d3d/d3d/B3'
#shot_dir='.'
#for current folder use '.'
#shot_no='170882.pq'

shot = 170882 # this is the shot number you're looking at
sh = DataPrep(shot) # this creates the DataPrep Object
properties = sh.peak_properties()
window_num = 1 # this is the specific window you're looking for. For a list of available window_num, do:
#print(properties.index.get_level_values(level=0))
window = properties.xs(window_num, level=0) # This retrieves only the data for the specified window_num
window['Widths'] = window['Left/Right'].apply(lambda x: [*map(lambda y: y[1]-y[0], x)]) #This adds a new column to the window dataframe containing a list of all the widths of the modes.

percentages = window.index.get_level_values(level='% ELM').tolist()
locations = window['Peak'].tolist()
widths = window['Widths'].tolist()


data_labeler(a,overlap_percent=50.)


a=DataPrep(shot_dir,shot_no)

a.elm_loc(plot=False,csv_output=True)#Locate the ELM

a.make_mask(plot=False,csv_output=True)


print(a)
