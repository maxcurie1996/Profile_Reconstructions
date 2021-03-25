import numpy as np 			#import numpy library and give it a short name as np
import sys as sys
sys.path.insert(1, '/u/mcurie/Scripts/tokaml-JeffsBranch/SULI2021/random_tools')
from DataPrep import *

#python /u/mcurie/Scripts/tokaml-JeffsBranch/0run_freq_find.py
#shot_dir='/p/datad2/gdong/signal_data_ELM/d3d/d3d/B3'
shot_dir='.'
#for current folder use '.'
shot_no='170882.pq'

a=DataPrep(shot_dir,shot_no)

a.elm_loc(plot=False,csv_output=True)#Locate the ELM

a.make_mask(plot=False,csv_output=True)


print(a)
