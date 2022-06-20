import tensorflow as tf
import math
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

class RecommenderNet(tf.keras.Model):
  
  # Insialisasi fungsi
  def __init__(self, num_users, num_tempat, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_tempat = num_tempat
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.tempat_embedding = layers.Embedding( # layer embeddings tempat
        num_tempat,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.tempat_bias = layers.Embedding(num_tempat, 1) # layer embedding tempat bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    tempat_vector = self.tempat_embedding(inputs[:, 1]) # memanggil layer embedding 3
    tempat_bias = self.tempat_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_tempat = tf.tensordot(user_vector, tempat_vector, 2) 
 
    x = dot_user_tempat + user_bias + tempat_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

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

      self.min_rating = None
      self.max_rating = None
      self.x_train    = None
      self.x_val      = None
      self.y_train    = None
      self.y_val      = None

      self.model = None
      self.tempat_to_tempat_encoded = None
      self.tempat_encoded_to_tempat = None
      self.user_to_user_encoded = None
      self.user_tempat_array = None
      self.tempat_not_visited = None
      self.tempat_visited_by_user = None

      self.user_id = None
      self.tempat_mydata = None

      self.num_users = None
      self.num_tempat = None

    def preprocessingdataTabel(self):
      # Mengubah ID_User menjadi list tanpa nilai yang sama
      user_ids = self.mydata['ID_User'].unique().tolist()
      # Melakukan encoding ID_User
      self.user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

      # Mengubah ID_Tempat menjadi list tanpa nilai yang sama
      tempat_ids = self.mydata['ID_Tempat'].unique().tolist()
      # Melakukan proses encoding ID_Tempat
      self.tempat_to_tempat_encoded = {x: i for i, x in enumerate(tempat_ids)}
      # Melakukan proses encoding angka ke ID_Tempat
      self.tempat_encoded_to_tempat = {i: x for i, x in enumerate(tempat_ids)}
      
      # Selanjutnya, petakan ID_User dan ID_Tempat ke dataframe yang berkaitan.
      # Mapping ID_User ke dataframe user
      self.mydata['Nama_User'] = self.mydata['ID_User'].map(self.user_to_user_encoded)
      # Mapping ID_Tempat ke dataframe tempat
      self.mydata['Nama_Tempat'] = self.mydata['ID_Tempat'].map(self.tempat_to_tempat_encoded)

      # Mendapatkan jumlah user
      self.num_users = len(self.user_to_user_encoded)
      print(self.num_users)
      # Mendapatkan jumlah tempat
      self.num_tempat = len(self.tempat_encoded_to_tempat)
      print(self.num_tempat)
      # Mengubah rating menjadi nilai float
      self.mydata['Rating'] = self.mydata['Rating'].values.astype(np.float32)
      
      # Nilai minimum rating
      self.min_rating = min(self.mydata['Rating'])
      # Nilai maksimal rating
      self.max_rating = max(self.mydata['Rating'])
      print('Number of User: {}, Number of Tempat: {}, Min Rating: {}, Max Rating: {}'.format(
          self.num_users, self.num_tempat, self.min_rating, self.max_rating
      ))

    def trainingdata(self):
      # Mengacak dataset
      mydata = self.mydata.sample(frac=1, random_state=42)
      x = mydata[['Nama_User', 'Nama_Tempat']].values
 
      # Membuat variabel y untuk membuat rating dari hasil 
      y = mydata['Rating'].apply(lambda x: (x - self.min_rating) / (self.max_rating - self.min_rating)).values
      
      # Membagi menjadi 80% data train dan 20% data validasi
      train_indices = int(0.8 * mydata.shape[0])
      self.x_train, self.x_val, self.y_train, self.y_val = (
          x[:train_indices],
          x[train_indices:],
          y[:train_indices],
          y[train_indices:]
      )

    def creatingUserArray(self):
      self.tempat_mydata = self.tempat_new
      data = pd.read_csv('Data Orbit - Data Gabungan.csv')
      mydata = data[1:814]

      # Mengambil sample user
      self.user_id = mydata['ID_User'].sample(100).iloc[30]
      self.tempat_visited_by_user = mydata[mydata['ID_User'] == self.user_id]
      
      # Operator bitwise (~)
      self.tempat_not_visited = self.tempat_mydata[~self.tempat_mydata['id'].isin(self.tempat_visited_by_user.ID_Tempat.values)]['id'] 
      self.tempat_not_visited = list(
          set(self.tempat_not_visited)
          .intersection(set(self.tempat_to_tempat_encoded.keys()))
      )
      
      self.tempat_not_visited = [[self.tempat_to_tempat_encoded.get(x)] for x in self.tempat_not_visited]
      user_encoder = self.user_to_user_encoded.get(self.user_id)
      self.user_tempat_array = np.hstack(
          ([[user_encoder]] * len(self.tempat_not_visited), self.tempat_not_visited)
      )

    def modelCreate(self):
      self.model = RecommenderNet(self.num_users, self.num_tempat, 50)
      self.model.compile(
          loss = tf.keras.losses.BinaryCrossentropy(),
          optimizer = keras.optimizers.Adam(learning_rate=0.001),
          metrics=[tf.keras.metrics.RootMeanSquaredError()]
      )
      self.model.fit(x = self.x_train,y = self.y_train,batch_size = 8,epochs = 100,validation_data = (self.x_val, self.y_val))

    def run_preprocessing(self):
      self.preprocessingdataTabel()
      self.trainingdata()
      self.modelCreate()
      self.creatingUserArray()
    
    def getData(self):
      rating = self.model.predict(self.user_tempat_array).flatten()
 
      top_rating_indices = rating.argsort()
      recommended_tempat_ids = [
          self.tempat_encoded_to_tempat.get(self.tempat_not_visited[x][0]) for x in top_rating_indices
      ]
      
      print('Menampilkan Rekomendasi  Tempat Makan dan Lokasi untuk Pengguna dengan Total : {} Pengguna'.format(self.user_id))
      print('===' * 11)
      top_tempat_user = (
          self.tempat_visited_by_user.sort_values(
              by = 'Rating',
              ascending=False
          )
          .head()
          .ID_Tempat.values
      )
      
      tempat_mydata_rows = self.tempat_mydata[self.tempat_mydata['id'].isin(top_tempat_user)]
      for row in tempat_mydata_rows.itertuples():
          print(row.nama, ':', row.lokasi, ':', row.gambar, ':', row.menu)
      
      print('---' * 11)
      print('10 Rekomendasi tempat makan')
      print('---' * 11)
      
      recommended_tempat = self.tempat_mydata[self.tempat_mydata['id'].isin(recommended_tempat_ids)]
      for row in recommended_tempat.itertuples():
          print(row.nama, ':', row.lokasi,':', row.gambar, ':', row.menu)

      filejson = self.tempat_mydata
      filejson.to_json("static/rekomendasi_restauran.json",orient = 'records')

    def createDetail():
      file = pd.read_json("static/rekomendasi_restauran.json")
      return file