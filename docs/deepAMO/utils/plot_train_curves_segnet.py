import re
from sys import argv, exit
import matplotlib.pyplot as plt

if len(argv) != 2:
    print "usage: plot_train_curves.py stdot.txt"
    exit(1)


stout_txt = argv[1]
train_loss = []
validation_loss = []
iter_num = []
bkg_dice_score = []
def_dice_score = []
with open(stout_txt) as file:
    iter = 0
    for line in file:
        if line.startswith('train'):
            #print "train_loss",float(line.strip().split(':')[1])
            train_loss.append(float(line.strip().split(':')[1]))
            iter = iter + 1
            iter_num.append(iter)
        if line.startswith('val'):
            if line.strip().split(' ')[1][0] == 'l':
                validation_loss.append(float(line.strip().split(':')[1]))
            elif line.strip().split(' ')[1][0] == 'D':
                bkg_dice = re.findall("\d+\.\d+",line.strip().split(':')[1])[0]
                #print
                defect_dice= re.findall("\d+\.\d+",line.strip().split(':')[1])[1]
                bkg_dice_score.append(float(bkg_dice))
                def_dice_score.append(float(defect_dice))

        #print "valid_loss",float(line.strip().split(':')[1])
        # if line.startswith('val Dice'):
        #     print float(line.strip().split(':')[1])

plt.plot(iter_num,train_loss)
plt.plot(iter_num,validation_loss)
plt.plot(iter_num,bkg_dice_score)
plt.plot(iter_num,def_dice_score)

plt.xlabel('Epoch number')
#plt.ylabel('Loss')
plt.legend(['train','validation','bkg_dice','defect_dice'])
plt.show()