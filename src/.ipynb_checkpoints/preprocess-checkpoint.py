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
    
    correction = 0.2
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('\\')[-1]  # Causion: this is for windows!
            #filename = source_path.split('/')[-1] # for Non-windows
            current_path = image_file + filename
            image = cv2.imread(current_path)
            images.append(image)

            measurement = float(line[3])
            if(i == 0): # center
                measurements.append(measurement)
            if(i == 1): # left
                measurements.append(measurement + correction) # turn to right
            if(i == 2): # right
                measurements.append(measurement - correction) # turn to left

        augmented_images, augmented_measurements = [], []
        # data augmentation
        for image, measurement in zip(images, measurements):
            augmented_images.append(image)
            augmented_measurements.append(measurement)
            augmented_images.append(cv2.flip(image,1)) # flip the image to produce more other side tuning
            augmented_measurements.append(-measurement) # reverse the steering angle

    
        
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    
    return X_train, y_train