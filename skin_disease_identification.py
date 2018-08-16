
#Image classification


#Loading images

import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

def get_filepaths(directory):
	file_paths = [] 
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
			file_paths.append(filepath)  

    return file_paths  

path = "Desktop\\image_classificaiton"
full_file_paths = get_filepaths(path)
                                                              
#transforming images



#setup a standard image size; 
STANDARD_SIZE = (300, 167)
def img_to_matrix(filename, verbose=False):
    
    img = Image.open(filename)
    
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
    
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

imgdir1= "Desktop/image_classification/"
imgdir2 = "Desktop/image_classification/classification/"

images1 = [imgdir1+ f for f in os.listdir(imgdir1)]
images2 = [imgdir2+ f for f in os.listdir(imgdir2)]

images = images1+images2

#Train images 
data = []
for image in images:
    img = img_to_matrix(image)
    img = flatten_image(img)
    data.append(img)

data = np.array(data)



pca = RandomizedPCA(n_components=5)
train_x = pca.fit_transform(data)

ones = np.ones(23)
zeros = np.zeros(33)

train_y = np.concatenate((ones,zeros))


knn = KNeighborsClassifier()
knn.fit(train_x, train_y)

 #Test images
img_dir_test = "image_classification/test/

images_test = [img_dir_test+ f for f in os.listdir(img_dir_test)]

data_test = []
for image in images_test:
    img = img_to_matrix(image)
    img = flatten_image(img)
    data_test.append(img)

test_x = pca.transform(data_test)

knn.predict(test_x)

pd.crosstab(train_y,knn.predict(train_x),rownames=['Act'],colnames=['Predicted'])
