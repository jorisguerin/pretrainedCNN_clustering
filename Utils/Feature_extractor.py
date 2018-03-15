import numpy as np

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input as prpc_vgg_res
from keras.applications.xception import preprocess_input as prpc_xce_inc

from keras.models import Model
import keras.applications.vgg16 as vgg16
import keras.applications.vgg19 as vgg19
import keras.applications.resnet50 as res
import keras.applications.inception_v3 as inc
import keras.applications.xception as xce

import pickle
from glob import glob
import os

class Feature_extractor:
	
	def __init__(self, dataset, cnn_architecture, layer):
		
		self.dataset_path = "./Data/%s/" % dataset
		self.dataset_im_path = self.dataset_path + "Images/"
		self.dataset_feat_path = self.dataset_path + "Features/%s_%s" % (cnn_architecture, layer)

		ext_file = open( self.dataset_path + "extension.txt", "r")
		self.extension = ext_file.read()
		ext_file.close()
		
		self.n_files = len(glob(self.dataset_im_path + "*" + self.extension))

		self.network_name = cnn_architecture
		self.layer_name   = layer
		
		self.get_network_characteristics()
		print("Feature extractor: %s // %s" % (self.network_name, self.layer_name))

	def get_network_characteristics(self):
		if self.network_name == "vgg16":
			base_model = vgg16.VGG16(weights='imagenet')
			self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)
			self.tgt_size = (224, 224)
			self.prpc_fct = prpc_vgg_res
			
		elif self.network_name == "vgg19":
			base_model = vgg19.VGG19(weights='imagenet')
			self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)        
			self.tgt_size = (224, 224)
			self.prpc_fct = prpc_vgg_res
			
		elif self.network_name == "resnet":
			base_model = res.ResNet50(weights='imagenet')
			self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)        
			self.tgt_size = (299, 299)
			self.prpc_fct = prpc_vgg_res
			
		elif self.network_name == "xception":
			base_model = xce.Xception(weights='imagenet')
			self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)        
			self.tgt_size = (299, 299)
			self.prpc_fct = prpc_xce_inc
			
		elif self.network_name == "inception":
			base_model = inc.InceptionV3(weights='imagenet')
			self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)
			self.tgt_size = (299, 299)
			self.prpc_fct = prpc_xce_inc
			
		else:
			print("Error: possible network names:\n-'vgg16'\n-'vgg19'\n-'resnet'\n-'xception'\n-'inception'")
			
	def get_PIL_image(self, image_name):
		file_name = self.dataset_im_path + image_name + self.extension
		
		return image.load_img(file_name, target_size = self.tgt_size)
	
	def get_arr_image(self, image_name):
		pil_im = self.get_PIL_image(image_name)
	
		return image.img_to_array(pil_im)
	
	def get_prpc_image(self, image_name):
		arr_im = self.get_arr_image(image_name)
		arr_im = np.expand_dims(arr_im, axis = 0)
		
		return self.prpc_fct(arr_im)
	
	def extract(self, image_name):
		prpc_im = self.get_prpc_image(image_name)
		
		return np.ndarray.flatten(self.model.predict(prpc_im))

	def extract_and_save_features(self):
		if not os.path.exists(self.dataset_feat_path):
			os.makedirs(self.dataset_feat_path)
		else:
			return
		
		print("extracting and saving features ... ")
		
		for im in range(self.n_files):
			if im % 100 == 0:
				print("    %d/%d" % (im, self.n_files))
			features = self.extract(str(im))
			feat_file = open(self.dataset_feat_path + "/%s.p" % im, "wb")
			pickle.dump(features, feat_file)
			feat_file.close()