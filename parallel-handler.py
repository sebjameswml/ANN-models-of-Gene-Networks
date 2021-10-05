import os
import numpy as np
import pickle

# this script should take a parameter range and number of repeats, creating a
# large number of trials, split into batches equal to number of cores, then
# send to batch.py to run at once.

# number of nodes (arg 2)
param1 = np.arange(15,16) #min,max,stepsize 0.1 and 0.003

# not used in this example (arg 3)
param2 = np.arange(15,16)

print ('param1 = {0};   param2 = {1}'.format(param1, param2))

repeats = 1

totaltrials = repeats*len(param1)*len(param2)
print ('totaltrials = {0}'.format(totaltrials))

trials = np.zeros((totaltrials,3)) # replace p with number of parameters + 1

cores = 10
print ('cores = {0}'.format(cores))

batches = int(np.floor(totaltrials/cores)+1)
print ('batches (floor(totaltrials/cores)+1) = {0}'.format(batches))

# create a large array, each row is a trial, with cols ID,param1,param2.
offset = 0
ID = offset

for x in range(repeats):
	for p1 in param1:
		for p2 in param2:
			trials[ID - offset] = [ID,p1,p2]
			ID = ID + 1

print('trials: {0}'.format(trials))

# compile the script just the once
scriptname = "main"
compile_cmd = "g++ --std=c++17 " + scriptname + ".cpp -o dans_genenet -I../morphologica -I/usr/include/hdf5/serial -I/usr/include/jsoncpp -Wl,-rpath,/usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/x86_64-linux-gnu/libjsoncpp.a -lpthread -lsz -lz -ldl -lm -O3 -Wall"
# if main.cpp or RNet.h are newer than dans_genenet, then re-compile
if (os.path.exists('dans_genenet')
        and (os.path.getctime('dans_genenet') < os.path.getmtime('main.cpp')
                 or os.path.getctime('dans_genenet') < os.path.getmtime('RNet.h'))) or not os.path.exists('dans_genenet'):

    print ('\nCompiling program with:\n   {0}'.format(compile_cmd))
    os.system (compile_cmd)
    print ('...done.\n')

else:
    print ('dans_genenet binary is up to date.\n')

count = 0
for i in range(batches):
	# take a batch of trials and save them for batch.py to access
	with open("batch.p", "wb") as f:
		pickle.dump(trials[count:count+cores], f) #if this goes out of range numpy will deal fine
	# run the batch
	os.system ("python batch.py")
	print ('Batch count = {0} is done.'.format(count))
    # python should wait for all cores to finish, although if some of them
    # don't finish and there's an overlap it shouldn't break, just slow down.
	count = count + cores
