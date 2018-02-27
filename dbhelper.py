import io
import numpy as np
import pandas as pd 
import sqlite3

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)



class DBHelper:
	def connect(self):
		return sqlite3.connect('cbir.db', detect_types=sqlite3.PARSE_DECLTYPES)

	def createSchema(self):
		'''Create the DB schema'''
		connection = self.connect()
		try:
			# images
			query = '''CREATE TABLE IF NOT EXISTS images (
			 img_hash TEXT PRIMARY KEY,
			 img_path TEXT NOT NULL,
			 img_arr array,
			 opened_image array,
			 color_vector array
			);'''
			cursor = connection.cursor()
			cursor.execute(query)
			connection.commit()

			# feature vectors
			query = '''CREATE TABLE IF NOT EXISTS feature_vectors (
			 img_hash TEXT,
			 method TEXT,
			 feature_vector array,
			 PRIMARY KEY (img_hash, method) 
			);'''
			cursor = connection.cursor()
			cursor.execute(query)
			connection.commit()	

		finally:
			connection.close()


	def deleteFeatures(self):
		'''Create images table in the DB schema'''
		connection = self.connect()
		try:
			query = '''DELETE FROM feature_vectors'''
			cursor = connection.cursor()
			cursor.execute(query)
			connection.commit()
		finally:
			connection.close()

	def addImages(self, images):
		'''Funciton that accepts a list of images and adds to to the database'''
		connection = self.connect()
		cursor = connection.cursor()

		try:
			for image in images:
				img_hash = image[0]
				img_path = image[1]
				img_arr = image[2]
				color_vector = image[3]
				opening = image[4]
				try:
					cursor.execute('INSERT INTO images VALUES (?,?,?,?,?)', (img_hash, img_path, img_arr, color_vector, opening))
				except: # primary key constraint
					pass
			connection.commit()
		finally:
			connection.close()


	def addFeatureVectors(self, image_features, method='sift'):
		'''Funciton that accepts a list of images and adds to to the database'''
		connection = self.connect()
		cursor = connection.cursor()

		try:
			for image in image_features:
				img_hash = image[0]
				feature_vector = image[1]
				feature_tuple = (img_hash, method, feature_vector)

				try:
					cursor.execute('INSERT INTO feature_vectors VALUES (?,?,?)', feature_tuple)
				except: # primary key constraint
					pass
			connection.commit()
		finally:
			connection.close()

	def getImages(self, use_opening=False):
		'''Function to return the images table as a DataFrame'''
		connection = self.connect()
		if use_opening==False:
			query = '''SELECT img_hash, img_arr FROM images'''
		else:
			query = '''SELECT img_hash, opened_image AS img_arr FROM images'''

		images_df = pd.read_sql_query(query, connection, chunksize=50)

		return images_df
		


	def collectFeatures(self):
		'''Function to collect feature vectors for the image library given a method'''
		query = """
				SELECT fv.img_hash, fv.feature_vector, img.color_vector
				FROM feature_vectors fv
				JOIN images img ON img.img_hash = fv.img_hash
				"""
		try:
			connection = self.connect()
			cursor = connection.cursor()
			cursor.execute(query)
			features = []
			for row in cursor:
				features.append(list(row))

			
			results = pd.DataFrame(features,columns=['img_hash','feature_vector','color_vector'])
			# split the images from the features
			images = np.asarray(results['img_hash'])
			features = np.array(list(results['feature_vector']))
			colors = np.array(list(results['color_vector']))

			return images, features, colors
			

		finally:
			connection.close()


	def retrieveMatches(self,matches):
		'''Function to collect feature vectors for the image library given a method'''
		query = """
				SELECT img_arr
				FROM images
				WHERE img_hash in (""" + matches + """)
				"""
		try:
			connection = self.connect()
			cursor = connection.cursor()
			cursor.execute(query)
			results = []
			for row in cursor:
				results.append(row[0])

			return results
			

		finally:
			connection.close()


	def retrieveImage(self,img_hash):
		'''Function to receive a hash for an image and return the image array'''
		query = """
				SELECT img_arr
				FROM images
				WHERE img_hash = '""" + img_hash + """'
				"""
		try:
			connection = self.connect()
			cursor = connection.cursor()
			cursor.execute(query)
			results = []
			for row in cursor:
				results.append(row[0])

			return results[0]
			

		finally:
			connection.close()


	def retrieveMethod(self):
		'''Function to receive a hash for an image and return the image array'''
		query = """
				SELECT method
				FROM feature_vectors
				LIMIT 1
				"""
		try:
			connection = self.connect()
			cursor = connection.cursor()
			cursor.execute(query)
			results = []
			for row in cursor:
				results.append(row[0])

			return results[0]
			

		finally:
			connection.close()