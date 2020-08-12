import numpy as np
import h5py

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

data = read_process_h5("train/digitStruct.mat")
