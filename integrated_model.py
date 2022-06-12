import tensorflow as tf
import math
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

class ProcessData():
    def __init__(self,csvFile):
      """Data yang di Inputkan merupakan nama file csv tempat dan rating user"""
      self.mydata   = pd.read_csv(csvFile) #File Utama
      self.mydata.columns = ["ID_Tempat", "Nama_Tempat", "ID_User", "Nama_User", "Daerah", "Latitude", "Longitude", "Rating", "Telepon"]
        
      self.mydata['Rating'] = self.mydata['Rating'].replace(r'\D+','',regex=True)
      self.mydata['Rating'] = pd.to_numeric(self.mydata['Rating'])
      self.mydata['Latitude'] = self.mydata['Latitude'].replace(r'\D+','',regex=True)
      self.mydata['Latitude'] = pd.to_numeric(self.mydata['Latitude'])
      self.mydata['Longitude'] = self.mydata['Longitude'].replace(r'\D+','',regex=True)
      self.mydata['Longitude'] = pd.to_numeric(self.mydata['Longitude'])

      df = pd.read_csv('Data Orbit - Data Tempat lengkap.csv')
      self.tempat_new = pd.DataFrame({
          'id': df.ID_Tempat,
          'nama': df.Nama_Tempat,
          'lokasi': df.Daerah,
          'gambar': df.Gambar,
          'menu': df.Menu
      })

      self.my_tf_saved_model = tf.keras.models.load_model('./saved_models/my_tf_model')

    def preprocessingdataTabel(self):
      # Mengubah ID_User menjadi list tanpa nilai yang sama
      user_ids = self.mydata['ID_User'].unique().tolist()
      # Melakukan encoding ID_User
      user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
      # Melakukan proses encoding angka ke ke ID_User
      user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

      # Mengubah ID_Tempat menjadi list tanpa nilai yang sama
      tempat_ids = self.mydata['ID_Tempat'].unique().tolist()
      # Melakukan proses encoding ID_Tempat
      tempat_to_tempat_encoded = {x: i for i, x in enumerate(tempat_ids)}
      # Melakukan proses encoding angka ke ID_Tempat
      tempat_encoded_to_tempat = {i: x for i, x in enumerate(tempat_ids)}
      
      # Selanjutnya, petakan ID_User dan ID_Tempat ke dataframe yang berkaitan.
      # Mapping ID_User ke dataframe user
      self.mydata['Nama_User'] = self.mydata['ID_User'].map(user_to_user_encoded)
      # Mapping ID_Tempat ke dataframe tempat
      self.mydata['Nama_Tempat'] = self.mydata['ID_Tempat'].map(tempat_to_tempat_encoded)

      # Mendapatkan jumlah user
      num_users = len(user_to_user_encoded)
      print(num_users)
      # Mendapatkan jumlah tempat
      num_tempat = len(tempat_encoded_to_tempat)
      print(num_tempat)
      # Mengubah rating menjadi nilai float
      self.mydata['Rating'] = self.mydata['Rating'].values.astype(np.float32)
      
      # Nilai minimum rating
      min_rating = min(self.mydata['Rating'])
      # Nilai maksimal rating
      max_rating = max(self.mydata['Rating'])
      print('Number of User: {}, Number of Tempat: {}, Min Rating: {}, Max Rating: {}'.format(
          num_users, num_tempat, min_rating, max_rating
      ))

    def run_preprocessing(self):
      print("hai")