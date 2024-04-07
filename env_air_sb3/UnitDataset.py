import os.path

import numpy as np
import os
from CreateDataset import Dataset
内存吃不住

whole = Dataset()
file_path = "/home/ming/aaa/AquaML-2.2.0/dataset/ExpertAirDocking"
for i in range(7, 13):
    partial = {'obs': [],
               'action': [],
               'total_reward': [],
               'next_obs': [],
               'mask': [],}
    for key in partial:
        # path = data_file_path + 'ExpertAirDocking' + str(i) + '/' + key + '.npy'
        path = os.path.join(file_path + str(i), key + '.npy')
        data = np.load(path)
        partial[key].extend(data)
    whole.add(partial)
whole.save('/home/ming/aaa/AquaML-2.2.0/dataset/ExpertAirDocking4000/')