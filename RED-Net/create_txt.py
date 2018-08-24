import os

def generate(dir):
    files = os.listdir(dir)
    print '****************'
    print 'input :',dir
    print 'start...'
    listText = open('train.txt','w')
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = 'train_noise/'+file +' '+'train_gt/'+file.split('_')[0]+'.jpg' +'\n'
        listText.write(name)
    listText.write('train_noise/'+file +' '+'train_gt/'+file.split('_')[0]+'.jpg')
    listText.close()
    print 'down!'
    print '****************'

if __name__ == '__main__':
    generate('/root/hj9/images/train_noise/')