import os
import numpy as np
import pickle

with open("batch.p", "rb") as f:
    batch = pickle.load(f)

# && fg keeps latest script in foreground so ctrl-C can close
command = ''
for row in batch:
    command = command + './dans_genenet ' + str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2])# + ' && fg'
    print ('batch.py: command is {0}'.format (command))
    os.system (command)
