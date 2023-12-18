import os
import glob
import string
import random
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import urllib.request

from flask import session
from keras import backend
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten
from keras.models import load_model
from keras.layers import Layer

from google.cloud import storage

#  service account cloud storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'serviceAccountKey_storage.json'

storage_client = storage.Client()


class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    

# with tf.keras.utils.custom_object_scope({'L1Dist':L1Dist}):
#     siamese_model = load_model('model/siamesemodelv2.h5')

# siamese_model = tf.keras.models.load_model('model/siamesemodelv2.h5')
siamese_model = tf.keras.models.load_model('model/siamese_model.h5',
                                                custom_objects={'L1Dist':L1Dist,
                                                    'BinaryCrossentropy':tf.losses.BinaryCrossentropy},
                                                    compile=False)


df = pd.DataFrame(columns=['model/siamese_model.h5', 'pred'])


path_store = 'static/images/stored_image'
path_input = 'static/images/input_image'
dir_path = r'static/images/stored_image/**/*.jpg*'
img_path = []
for file in glob.glob(dir_path, recursive=True):
    img_path.append(file)

def get_random_string(length):
    # random string dengan kombinasi upper dan lower case
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

def pair_list(input_file_name):
    bucket_name = 'lokana'
    dir_inp = f'https://storage.googleapis.com/{bucket_name}/input_image/{input_file_name}'  # Path dari GCS
    stored_img_path = []
    input_img_path = []
    
    # Ambil path dari GCS untuk stored images
    blobs = list(storage_client.list_blobs(bucket_name, prefix='stored_image/'))
    for blob in blobs:
        stored_img_path.append(f'https://storage.googleapis.com/{bucket_name}/{blob.name}')  
        input_img_path.append(dir_inp) 
    return stored_img_path, input_img_path


def prep(path):
    # Mengambil gambar dari GCS
    req = urllib.request.urlopen(path)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)  # Membaca gambar dari URL
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (250,250))
    image = tf.expand_dims(image, axis=0)
    return image

def preds(input):
    y_pred = siamese_model.predict(input)
    return y_pred[0][0]

def pred_image(data, img_file_path):
    input_dir = img_file_path
    file_dir  = data['file_path']
    y_pred = []
    
    #membaca gambar input/yang ingin ditest karena semua input sama maka tidak perlu dilooping
    im_test = prep(input_dir)
    for i in range(len(file_dir)):
        #membaca gambar stored
        im_strd = prep(file_dir[i])
        #membuat list untuk dimasukan ke dua input
        im = [im_test,im_strd]
        pred = preds(im)
        y_pred.append(pred)
    print(y_pred)    
    return y_pred

def visualize(data):
    data = df.loc[df['pred'] >= 0.5]
    inp = df['input_path'].iloc[0]
    plt.figure(figsize = (5,5))
    inp_im = cv2.imread(inp)
    inp_im = cv2.cvtColor(inp_im, cv2.COLOR_BGR2RGB)
    plt.imshow(inp_im)
    plt.title(label="Input Image",fontsize=14)
    plt.show()
    for i in range(len(data)):
        #membuat variabel path stored image
        strd = data['file_path'].iloc[i]
        #membuat variabel nama image
        fl_name = data['file_path'].iloc[i].split("\\")[3]
        #membuat variabel nilai preediksi
        pred = round(float(data['pred'].iloc[0]),3)
        
        #membuat plot gambar
        plt.figure(figsize = (5,5))
        strd_im = cv2.imread(strd)
        strd_im = cv2.cvtColor(strd_im, cv2.COLOR_BGR2RGB)
        plt.imshow(strd_im)
        plt.title(label=f"{fl_name} have {pred} similarity with Input Image",fontsize=14)
        plt.show()
        
        
# Fungsi endpoint
# Fungsi untuk menghapus foto dari Google Cloud Storage
def delete_gcs_folder(bucket_name, folder_path):
    try:
        bucket = storage.Client().bucket(bucket_name)

        # Daftar semua blobs dalam folder
        blobs = bucket.list_blobs(prefix=folder_path)
        
        # Hapus setiap blob dalam folder
        for blob in blobs:
            blob.delete()

        # Hapus folder itu sendiri
        bucket.delete_blob(folder_path)

        return True
    except Exception as e:
        print(f"Error menghapus folder {folder_path} di bucket {bucket_name}: {str(e)}")
        return False

def delete_gcs_photo(photo_paths):
    for photo_path in photo_paths:
        if photo_path.startswith('gs://'):
            bucket_name, blob_path = photo_path[len('gs://'):].split('/', 1)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.delete()
            
counter = 1
def generate_id():
    global counter
    new_id = f"{counter:02}"  # Format ID dengan 2 digit, contohnya: "01", "02", dst.
    
    if counter == 9:
        counter += 1  # Menambah counter
        return f"{counter:03}"  # Format ID dengan 3 digit setelah "09"
    
    counter += 1
    return new_id            