from scipy import misc
import numpy as np
import cv2
image_size = (54,128)
import h5py

def createImageData(datapoint, output_size, path):
    rawimg = cv2.imread(path+datapoint['filename'])
    img = np.array(cv2.resize(rawimg, output_size))
    return img, rawimg.shape

def generateData(data, n=1000):
    '''
    generates flattened SVHN dataset
    '''

    Ximg_flat = []
    Xidx_flat = []
    ycount_flat = []
    ycoord_flat = []
    ylabel_flat = []
    
    for datapoint in np.random.choice(data, size=n, replace=False):
        print(datapoint)
        img,_ = createImageData(datapoint, image_size, 'train/')
        for i in range(0,datapoint['length']):
            Ximg_flat.append(img)
            Xidx_flat.append(i)
            ycount_flat.append(datapoint['length'])
            ylabel_flat.append(datapoint['labels'][i])
            
    ylabel_flat = [0 if y==10 else int(y) for y in ylabel_flat]
    return np.array(Ximg_flat), np.array(Xidx_flat), np.array(ycount_flat), np.array(ylabel_flat)

def read_process_h5(filename):
    """ Reads and processes the mat files provided in the SVHN dataset. 
        Input: filename 
        Ouptut: list of python dictionaries 
    """ 
        
    f = h5py.File(filename, 'r')

    groups = f['digitStruct'].keys()

    bbox_ds = np.array(f['digitStruct/bbox']).squeeze()
    names_ds =np.array(f['digitStruct/name']).squeeze()

    data_list = []
    num_files = bbox_ds.shape[0]
    count = 0

    for objref1, objref2 in zip(bbox_ds, names_ds):

        data_dict = {}

        # Extract image name
        names_ds = np.array(f[objref2]).squeeze()
        filename = ''.join(chr(x) for x in names_ds)
        data_dict['filename'] = filename

        #print filename

        # Extract other properties
        items1 = f[objref1].items()
      #  print(f[objref1].keys())
        # Extract image label
        labels_ds = np.array(f[objref1]['label']).squeeze()
        try:
            label_vals = [int(f[ref][:][0, 0]) for ref in labels_ds]
        except TypeError:
            label_vals = [labels_ds]
        data_dict['labels'] = label_vals
        data_dict['length'] = len(label_vals)

        # Extract image height
        height_ds = np.array(f[objref1]['height']).squeeze()
        try:
            height_vals = [f[ref][:][0, 0] for ref in height_ds]
        except TypeError:
            height_vals = [height_ds]
        data_dict['height'] = height_vals

        # Extract image left coords
        left_ds = np.array(f[objref1]['left']).squeeze()
        try:
            left_vals = [f[ref][:][0, 0] for ref in left_ds]
        except TypeError:
            left_vals = [left_ds]
        data_dict['left'] = left_vals

        # Extract image top coords
        top_ds = np.array(f[objref1]['top']).squeeze()
        try:
            top_vals = [f[ref][:][0, 0] for ref in top_ds]
        except TypeError:
            top_vals = [top_ds]
        data_dict['top'] = top_vals

        # Extract image width
        width_ds = np.array(f[objref1]['width']).squeeze()
        try:
            width_vals = [f[ref][:][0, 0] for ref in width_ds]
        except TypeError:
            width_vals = [width_ds]
        data_dict['width'] = width_vals

        data_list.append(data_dict)

        count += 1
        #print ("Processed:" +count +" ," + num_files)

    return data_list

