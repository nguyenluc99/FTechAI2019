import os, os.path

# path joining version for other paths
DIR = '/home/lucnguyen/Documents/FTechAI/image'
lst = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
print('initial lst , len {} is : {}, {}'.format(len(lst), lst, '\n'))
for i in range(0, len(lst)) :
    src = DIR + '/' + lst[i]
    dest = DIR + '/' + str(i)
    os.rename(src, dest)
lst2 = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
print('new lst2 , len {} is : {}, {}'.format(len(lst2), lst2, '\n'))

