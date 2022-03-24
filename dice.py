import numpy as np

k=1

# segmentation
seg = np.zeros((100,100, 100), dtype='int')
seg[30:70, 30:70, 30:70] = k

# gt
gt = np.zeros((100,100, 100), dtype='int')
gt[30:70, 40:80, 30:70] = k

dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))

print('Dice score is {0}'.format(dice))
