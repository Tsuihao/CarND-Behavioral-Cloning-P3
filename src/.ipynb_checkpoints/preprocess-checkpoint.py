import csv
import cv2
import numpy as np

def get_train_data_from_sim_data(csv_file, image_file):
    """
    csvfile:
    imagefile:
    
    return: (in np array)
    X_train
    y_train
    """
    lines = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    
    for line in lines:
        source_path = line[0]
        filename = source_path.split('\\')[-1]  # Causion: this is for windows!
        #filename = source_path.split('/')[-1] # for Non-windows
        current_path = image_file + filename
        image = cv2.imread(current_path)
        images.append(image)
        
        measurement = float(line[3])
        measurements.append(measurement)
        
    X_train = np.array(images)
    y_train = np.array(measurements)
    
    return X_train, y_train
        