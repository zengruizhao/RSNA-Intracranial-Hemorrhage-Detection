import os
import pydicom
import matplotlib.pylab as plt
import numpy as np

path = '/media/tiger/zzr/rsna'
trainPath = os.path.join(path, 'test')

def preprocess(img):
    ds = pydicom.dcmread(img)
    try:
        windowCenter = int(ds.WindowCenter[0])
        windowWidth = int(ds.WindowWidth[0])
    except:
        windowCenter = int(ds.WindowCenter)
        windowWidth = int(ds.WindowWidth)
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    data = ds.pixel_array
    data = np.clip(data * slope + intercept, windowCenter - windowWidth/2, windowCenter + windowWidth/2)
    plt.imshow(data, cmap='gray')
    plt.show()


if __name__ == '__main__':
    dataList = sorted(os.listdir(trainPath))
    for i in dataList:
        print(i)
        img = os.path.join(trainPath, i)
        preprocess(img)