#! /usr/bin/env python3

import math
import sys

class NBC(object):

	def __init__(self, nclasses, verbose, ):
		self._nclasses = nclasses
		self._verbose = verbose
		self._training = None
		self._test = None

	def train(self, training):
		self._training = training
		self._relation_training = ""
		self._attributes_training = []
		self._N = {}
		self._total_training = -1

		with open(self._training, 'r') as dataset:
			self._total_training = 0

			data_mode = False
			for line in dataset:
				line = self._remove_comments(line)

				# parse header
				if not data_mode:
					if line.startswith("@relation"):
						self._relation_training = line[line.find(" ")+1:]
					elif line.startswith("@attribute"):
						tokens = line.split()
						name = tokens[1]
						domain = [ v for v in tokens[2][1:-1].split(",") ]
						self._attributes_training.append((name,domain))
					elif line.startswith("@data"):
						data_mode = True

				# parse data
				else:
					self._total_training += 1
					instance = line.split(",")
					attributes = instance[:-self._nclasses]
					classes = instance[-self._nclasses:]
					assert(len(instance) == len(self._attributes_training))
					assert(len(attributes) + len(classes) == len(self._attributes_training))
					for i in range(len(attributes)):
						a = attributes[i]
						assert(a in self._attributes_training[i][1])
						for j in range(len(classes)):
							c = classes[j]
							self._N[ ( (i,a), (j,c) ) ] = self._N.get(( (i,a), (j,c) ), 0) + 1
					for j in range(len(classes)):
						c = classes[j]
						assert(c in ['0','1'])
						self._N[ (j,c) ] = self._N.get( (j,c), 0) + 1

			self._classes = [ attr[0] for attr in self._attributes_training[-self._nclasses:] ]

		if self._verbose > 0:
			print("=== TRAINING ===")
			print()
			print(">> training dataset: {0}".format(self._training))
			print(">> number of attributes = {0}".format(len(self._attributes_training)))
			print(">> number of training instances = {0}\n".format(self._total_training))
			if self._verbose > 1:
				print("@relation = {0}\n".format(self._relation_training))
				maxlen = max([len(attr[0]) for attr in self._attributes_training])
				print("@attributes = {")
				for name,domain in self._attributes_training[:-self._nclasses]:
					print( "  {0:{maxlen}}  :   domain={1}".format(name,domain,maxlen=maxlen))
				print("}\n")
				print("@classes = {")
				for name,domain in self._attributes_training[-self._nclasses:]:
					print( "  {0:{maxlen}}  :   domain={1}".format(name,domain,maxlen=maxlen))
				print("}\n")
			if self._verbose > 2:
				print("@counts = {")
				for key,val in self._N.items():
					print("  N[{0}] = {1}".format(key,val))
				print("}")
				print()

	def test(self, test):
		assert(self._training is not None)
		self._test = test
		self._attributes_test = []

		self._right = {}
		self._wrong = {}

		self._total_test = 0
		data_mode = False
		with open(self._test, 'r') as dataset:
			for line in dataset:
				line = self._remove_comments(line)

				# parse header
				if not data_mode:
					if line.startswith("@relation"):
						self._relation = line[line.find(" ")+1:]
					elif line.startswith("@attribute"):
						tokens = line.split()
						name = tokens[1]
						domain = [ v for v in tokens[2][1:-1].split(",") ]
						self._attributes_test.append((name,domain))
					elif line.startswith("@data"):
						data_mode = True

				# parse data
				else:
					self._total_test += 1
					instance = line.split(",")
					attributes = instance[:-self._nclasses]
					classes = instance[-self._nclasses:]
					assert(len(instance) == len(self._attributes_test))
					assert(len(instance) == len(self._attributes_training))
					assert(len(attributes) + len(classes) == len(instance))
					results = self._classify(attributes, classes)
					for j in range(len(classes)):
						class_name = self._classes[j]
						c = classes[j]
						if results[j] == c:
							self._right[class_name] = self._right.get(class_name,0) + 1
						else:
							self._wrong[class_name] = self._wrong.get(class_name,0) + 1

		if self._verbose > 0:
			print("=== TEST ===")
			print()
			print(">> test dataset: {0}".format(self._test))
			print(">> number of attributes = {0}".format(len(self._attributes_test)))
			print(">> number of test instances = {0}\n".format(self._total_test))
		print(">> results:")
		max_name_size = max([len(c) for c in self._classes])
		for j in range(len(classes)):
			name = self._classes[j]
			correct = self._right[name]
			incorrect = self._wrong[name]
			print("class = {0:{sz}} => correct = {1}, incorrect = {2}, ratio = {3:.4f}".format(name,correct,incorrect, correct/(correct + incorrect),sz=max_name_size))

	def _classify(self, attributes, classes):
		results = []
		n = len(attributes)
		for j in range(len(classes)):
			domain = self._attributes_test[-self._nclasses+j][1]
			max_ll = -sys.maxsize
			clss = None
			for v in domain:
				ll = (1-n)*math.log(self._N.get((j,v),len(domain)))
				for i in range(n):
					a = attributes[i]
					ll += math.log(self._N.get(((i,a),(j,v)),1))
				if ll > max_ll:
					clss = v
					max_ll = ll
			results.append(clss)
		return results

	def _remove_comments(self, line):
		comment = line.find("%")
		if comment:
			line = line[:comment]
		return line

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("classifier", choices=["nbc","tan","aod"], help="classifier type")
	parser.add_argument("training", help="path to training dataset")
	parser.add_argument("test", help="path to test dataset")
	parser.add_argument("-c", "--classes", type=int, default=1, help="number of class attributes (multidimensional)")
	parser.add_argument("-v", "--verbose", type=int, default=0, choices=[0,1,2,3], help="verbose mode (default 0)")
	args = parser.parse_args()

	nbc = NBC(args.classes, args.verbose)
	nbc.train(args.training)
	nbc.test(args.test)
