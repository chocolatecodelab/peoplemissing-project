import uuid
import string
import os
import random
import pathlib
import cv2
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import heapq

from keras import backend
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten
from io import BytesIO
from flask import Flask, render_template, request, jsonify, redirect, session, send_file
# from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from fungsi import get_random_string, prep, preds, pair_list, pred_image, visualize, delete_gcs_folder, delete_gcs_photo, generate_id
from keras.layers import Layer
from keras.models import load_model

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from google.cloud import storage

#  service account cloud storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'serviceAccountKey_storage.json'

storage_client = storage.Client()

# create only and running only one time
# bucket_name = 'seek-out'
# bucket = storage_client.bucket(bucket_name)
# bucket.location = 'ASIA'
# storage_client.create_bucket(bucket)

# get bucket
# my_bucket = storage_client.get_bucket('seek-out')



# service account firebase
cred = credentials.Certificate("serviceAccountKey.json")

# init 
app = firebase_admin.initialize_app(cred)
db = firestore.client()

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    


with tf.keras.utils.custom_object_scope({'L1Dist':L1Dist}):
    siamese_model = load_model('model/siamese_model.h5', compile=False)

siamese_model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

# load model
# siamese_model = tf.keras.models.load_model('model/siamesemodelv2.h5',
#                                                custom_objects={'L1Dist':L1Dist,
#                                                    'BinaryCrossentropy':tf.losses.BinaryCrossentropy},
#                                                    compile=False)

# configure path
app   = Flask(__name__, static_url_path='/static')
# app.config['UPLOAD_FOLDER'] = 'static/images/stored_image'
# app.config['UPLOADED_FILES'] = 'static/images/input_image'
    
@app.route('/', methods=['GET']) 
def index():
    try:
        return jsonify({"success": "Hello, World"}), 200
    except Exception as e: 
        return jsonify({"error": str(e)}), 500  
    
    
# insert (Create)
@app.route('/addpeople', methods=['POST'])
def add_people():
    try:
        if request.method == 'POST':
            # request photo
            fotos = request.files.getlist('fotos[]')
            # Memeriksa apakah daftar file foto kosong atau tidak ada
            for foto in fotos:
                if foto.filename == '':
                    return jsonify({'error': 'One or more photos are empty', 'status': '400'}), 400
            nama = request.form.get('nama')
            nama_with_underscore = nama.replace(' ', '_')
            foto_paths = []  # Inisialisasi array untuk menyimpan path setiap foto
            url_foto = []  # Inisialisasi array untuk menyimpan url setiap foto
            for foto in fotos:
                filename = secure_filename(foto.filename)
                ext = os.path.splitext(filename)[1]
                new_filename = get_random_string(20)

                bucket_name = 'seek-out'
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(f'stored_image/{nama_with_underscore}/{new_filename+ext}')
                
                # Upload foto langsung dari data yang diterima
                blob.upload_from_string(
                    foto.read(),  # Membaca data foto dari request
                    content_type=foto.content_type  # Menambahkan tipe konten untuk foto
                )
                
                file_path = f'gs://{bucket_name}/stored_image/{nama_with_underscore}/{new_filename+ext}'
                foto_paths.append(file_path)
                
                file_url_path = f'https://storage.googleapis.com/{bucket_name}/stored_image/{nama_with_underscore}/{new_filename+ext}'
                url_foto.append(file_url_path)

            # retrieve the data from data form
            nama = request.form['nama']
            umur = request.form['umur']
            tinggi = request.form['tinggi']
            berat_badan = request.form['berat_badan']
            ciri_fisik = request.form['ciri_fisik']
            nomor_dihubungi = request.form['nomor_dihubungi']
            sering_ditemukan_di = request.form['sering_ditemukan_di']
            kota = request.form['kota']
            gender = request.form['gender']
            isFound = request.form['isFound']

            expected_data = ['nama', 'umur', 'tinggi', 'berat_badan', 'ciri_fisik', 
                     'nomor_dihubungi', 'sering_ditemukan_di', 'kota', 
                     'gender', 'isFound']
    
            # Memeriksa keberadaan semua data yang diharapkan dalam permintaan form
            for data_key in expected_data:
                if data_key not in request.form or not request.form[data_key]:
                    return jsonify({'error': f'Missing or empty data: {data_key}', 'status': '400'}), 400

            # create id random
            new_id = generate_id()

            # Buat dokumen baru di koleksi 'people'
            addMissingPeople = {
                "id_people" : new_id,
                "foto": foto_paths,
                "url_foto": url_foto,
                "nama": nama,
                "umur": umur,
                "tinggi": tinggi,
                "berat_badan": berat_badan,
                "ciri_fisik": ciri_fisik,
                "nomor_dihubungi": nomor_dihubungi,
                "sering_ditemukan_di": sering_ditemukan_di,
                'kota': kota,
                'gender': gender,
                "isFound": isFound
            }
            
            db.collection('MissingPersons').add(addMissingPeople)


            response = {
                "status": "200",
                "message": "Person added successfully",
                "data": addMissingPeople
            }

            return jsonify(response), 200
    except Exception as e:
        response = {
            "status": "500",
            "message": str(e)
        }
        return jsonify(response), 500


# find people to compare 
@app.route('/findpeople', methods=['GET', 'POST'])
def findpeople():
    try:
        if request.method == 'POST':
            uploaded_img = request.files['uploaded_img']
            if not uploaded_img:
                return jsonify({'error': 'No photos found in the request', 'status': '400'}), 400
            img_filename = secure_filename(uploaded_img.filename)
            # no save to local
            # uploaded_img.save(os.path.join(app.config['UPLOADED_FILES'], img_filename))
            
            # for compare
            # img_file_path = os.path.join(app.config['UPLOADED_FILES'], img_filename) 
            # file_path = img_file_path
            
            # but insert to bucket cloud storage
            bucket_name = 'seek-out'
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(f'input_image/{img_filename}')
            
            # save to bucket
            # Upload foto langsung dari data yang diterima
            blob.upload_from_string(
                uploaded_img.read(),  # Membaca data foto dari request
                content_type=uploaded_img.content_type  # Menambahkan tipe konten untuk foto
            )
            file_path = f'gs://{bucket_name}/input_image/{img_filename}'
            file_url_path = f'https://storage.googleapis.com/{bucket_name}/input_image/{img_filename}'

            
            # add path image to firestore
            userFoto = {"foto" : file_path, "url_foto" : file_url_path}
            db.collection("UserSubmittedPhotos").add(userFoto) 
            
            # assign to the scheme pandas
            strd_pth, inp_pth = pair_list(img_filename) #return two list
            df = pd.DataFrame(list(zip(inp_pth, strd_pth)),columns =['input_path', 'file_path'])
            
            # # # Predict -> filter the photos result above average (>= 0.5)
            y_pred = pred_image(df, file_url_path)
            df['pred'] = y_pred
            dd = df.loc[df['pred'] >= 0.5]
            # # Maksimal 5 Photo (limit 5 photos)
            n = []
            for i in range(len(dd)):
                n.append(dd['pred'].iloc[i])
            
            heapq.heapify(n)
            pred5 = heapq.nlargest(5, n)

            #  Filtering and return to the array photos
            strd = []
            for i in range(len(dd)):
                if dd['pred'].iloc[i] in pred5:
                    strd.append(dd['file_path'].iloc[i])  
                       
            # koneksi database firestore 
            MissingPersons = db.collection("MissingPersons") 
            query = MissingPersons.where("url_foto", "array_contains_any", strd).stream() #array compare field url_foto and each strd index
            # Lakukan iterasi pada hasil query untuk menampilkan dokumen yang memenuhi kondisi
            # Buat daftar untuk menyimpan data dokumen yang cocok
            matched_documents = []
            # Iterasi pada hasil query dan ambil data yang diperlukan dari setiap dokumen
            for doc in query:
                matched_documents.append(doc.to_dict())  # Menambahkan data dokumen ke dalam daftar

            # jika isinya tidak ada
            if not matched_documents:
                return jsonify({"error": "Data tidak ditemukan", "status" : "400"}), 404  # Atau kode status yang sesuai
            # Return daftar yang berisi data dokumen yang cocok sebagai respons Flask
            response = {
                "status": "200",
                "message": "Searching person successfully",
                "data": matched_documents
            }
            return jsonify(response), 200
            
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500    


# Get people by criteria
@app.route('/getpeople', methods=['GET'])
# /getpeople?nama=John   
# /getpeople?kota=Jakarta
# /getpeople?gender=Male
# /getpeople?nama=John&kota=Jakarta&gender=Male 
# /getpeople?id_people=id_people 
def get_people_by_criteria():
    try:
        # Mendapatkan parameter dari URL
        nama = request.args.get('nama')
        kota = request.args.get('kota')
        gender = request.args.get('gender')
        id_people = request.args.get('id_people')

        # Inisialisasi query tanpa filter
        people_query = db.collection('MissingPersons')

        # Menambahkan filter berdasarkan nama jika parameter nama diberikan
        if nama:
            people_query = people_query.where('nama', '==', nama)

        # Menambahkan filter berdasarkan kota jika parameter kota diberikan
        if kota:
            people_query = people_query.where('kota', '==', kota)

        # Menambahkan filter berdasarkan gender jika parameter gender diberikan
        if gender:
            people_query = people_query.where('gender', '==', gender)

        # Menambahkan filter berdasarkan id_people jika parameter id_people diberikan
        if id_people:
            people_query = people_query.where('id_people', '==', id_people)
        # Menjalankan query
        people_collection = people_query.stream()

        # Membuat daftar untuk menyimpan data orang
        people_list = []

        # Mengambil data orang dari hasil query
        for person in people_collection:
            person_data = person.to_dict()
            people_list.append(person_data)

        if not people_list:  # Memeriksa apakah list kosong
            return jsonify({'error': 'people has not found', 'status': '400'}), 400

        response = {
            "status": "200",
            "message": "Data retrieved successfully",
            "data": people_list
        }

        return jsonify(response), 200
    except Exception as e:
        response = {
            "status": "400",
            "message": str(e)
        }
        return jsonify(response), 400

# edit person based on name
#PR for edit -> if user edit photo, then replace another photo, previous photo must be deleted
@app.route('/editpeople/<id>', methods=['PUT'])
def edit_people_by_name(id):
    try:
        # Periksa apakah orang dengan nama tertentu ada
        query = db.collection('MissingPersons').where('id_people', '==', id)
        results = query.stream()
        for doc in results:
            person_data = doc.to_dict()
            person_name = person_data.get('nama')
            print(person_name)
            person_ref = doc.reference

        if not person_ref.get().exists:
            return jsonify({'error': 'Person not found'}), 404
        
        nama_with_underscore = person_name.replace(' ', '_')
        # request foto
        fotos = request.files.getlist('fotos[]')

        for foto in fotos:
            if foto.filename == '':
                return jsonify({'error': 'One or more photos are empty', 'status': '400'}), 400

        foto_paths = []  # Inisialisasi array untuk menyimpan path setiap foto
        url_foto = []  # Inisialisasi array untuk menyimpan path setiap url foto

        # Menghapus foto sebelumnya jika ada
        delete_gcs_photo(person_ref.get().to_dict().get('foto', []))

        # Loop through new photos
        for foto in fotos:
            filename = secure_filename(foto.filename)
            ext = os.path.splitext(filename)[1]
            new_filename = get_random_string(20)

            # but insert to bucket cloud storage
            bucket_name = 'seek-out'
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(f'stored_image/{nama_with_underscore}/{new_filename+ext}')

            # Upload foto langsung dari data yang diterima
            blob.upload_from_string(
                foto.read(),  # Membaca data foto dari request
                content_type=foto.content_type  # Menambahkan tipe konten untuk foto
            )

            file_path = f'gs://{bucket_name}/stored_image/{nama_with_underscore}/{new_filename+ext}'
            foto_paths.append(file_path)
            file_url_path = f'https://storage.googleapis.com/{bucket_name}/stored_image/{nama_with_underscore}/{new_filename+ext}'
            url_foto.append(file_url_path)

        expected_data = ['nama', 'umur', 'tinggi', 'berat_badan', 'ciri_fisik', 
                 'nomor_dihubungi', 'sering_ditemukan_di', 'kota', 
                 'gender', 'isFound']

        # Memeriksa keberadaan semua data yang diharapkan dalam permintaan form
        for data_key in expected_data:
            if data_key not in request.form or not request.form[data_key]:
                return jsonify({'error': f'Missing or empty data: {data_key}', 'status': '400'}), 400
  

        # Ambil data dari form
        nama = request.form.get('nama', '')
        umur = request.form.get('umur', '')
        tinggi = request.form.get('tinggi', '')
        berat_badan = request.form.get('berat_badan', '')
        ciri_fisik = request.form.get('ciri_fisik', '')
        nomor_dihubungi = request.form.get('nomor_dihubungi', '')
        sering_ditemukan_di = request.form.get('sering_ditemukan_di', '')
        kota = request.form.get('kota', '')
        gender = request.form.get('gender', '')
        isFound = request.form.get('isFound', '')

        # Update data orang
        person_ref.update({
            'foto': foto_paths,
            'url_foto': url_foto,
            'nama': nama,
            'umur': umur,
            'tinggi': tinggi,
            'berat_badan': berat_badan,
            'ciri_fisik': ciri_fisik,
            'nomor_dihubungi': nomor_dihubungi,
            'sering_ditemukan_di': sering_ditemukan_di,
            'kota': kota,
            'gender': gender,
            'isFound': isFound
        })

        # Mendapatkan data yang sudah diupdate
        updated_data = person_ref.get().to_dict()

        response = {
            'status': '200',
            'message': f'Person with name {person_name} updated successfully',
            'updated_data': updated_data
        }

        return jsonify(response), 200
    except Exception as e:
        response = {
            'status': '500',
            'message': str(e)
        }
        return jsonify(response), 500


# delete person based on name
# PR for delete -> delete photos in cloud storage
@app.route('/deletepeople/<id>', methods=['DELETE'])
def delete_people_by_name(id):
    try:
        # Periksa apakah orang dengan nama tertentu ada
        query = db.collection('MissingPersons').where('id_people', '==', id)
        results = query.stream()
        for doc in results:
            person_data = doc.to_dict()
            person_name = person_data.get('nama')
            person_ref = doc.reference
            print("hello world pertama")
            # Menghapus foto dari Google Cloud Storage sebelum menghapus data orang
            delete_gcs_photo(person_ref.get().to_dict().get('foto', []))

            print("hello world")
            # Ekstrak jalur folder dari URL foto pertama (asumsi semua foto ada dalam folder yang sama)
            folder_path = doc.to_dict().get('foto', [])[0].split('/')[-2]
            print("hello world baru")
            # Hapus seluruh folder dari Google Cloud Storage
            delete_gcs_folder('seek-out', folder_path)

            # Hapus dokumen dari Firestore
            doc.reference.delete()

        response = {
            'status': '200',
            'message': f'Person dengan nama {person_name} berhasil dihapus'
        }

        return jsonify(response), 200
    except Exception as e:
        response = {
            'status': '500',
            'message': str(e)
        }
        return jsonify(response), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

