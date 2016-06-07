import math
import sys

class AOD(object):

	def __init__(self, nclasses, verbose, threshold):
		self._nclasses = nclasses
		self._verbose = verbose
		self._threshold = threshold
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
				line = line.replace("\n","")
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
						a_i = attributes[i]
						assert(a_i in self._attributes_training[i][1])
						for k in range(len(classes)):
							c = classes[k]
							self._N[ ( (i,a_i), (k,c) ) ] = self._N.get(( (i,a_i), (k,c) ), 0) + 1

							for j in range(len(attributes)):
								if j == i:
									continue
								a_j = attributes[j]
								assert(a_j in self._attributes_training[j][1])
								self._N[ ( (j,a_j), (i,a_i), (k,c) ) ] = self._N.get(( (j,a_j), (i,a_i), (k,c) ), 0) + 1

			self._classes = [ attr[0] for attr in self._attributes_training[-self._nclasses:] ]

		if self._verbose > 0:
			attributes = self._attributes_training[:-self._nclasses]
			classes = self._attributes_training[-self._nclasses:]
			max_domain = max([len(domain) for name,domain in attributes])
			avg_domain = sum([len(domain) for name,domain in attributes])/len(attributes)

			print("=== TRAINING ===")
			print()
			print(">> training dataset: {0}".format(self._training))
			print(">> number of instances  = {0}".format(self._total_training))
			print(">> number of fields = {0}".format(len(self._attributes_training)))
			print(">> number of classes    = {0}".format(len(classes)))
			print(">> number of attributes = {0}".format(len(attributes)))
			print(">> domain size: max = {0}, avg = {1}\n".format(max_domain, avg_domain))

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
				line = line.replace("\n","")
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
					results = self._classify(attributes)
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
			print(">> number of instances = {0}".format(self._total_test))
			print(">> number of fields = {0}\n".format(len(self._attributes_test)))

		print(">> Results:")
		max_name_size = max([len(c) for c in self._classes])
		total_correct = 0
		total_incorrect = 0
		for j in range(len(self._classes)):
			name = self._classes[j]
			correct = self._right.get(name,0)
			total_correct += correct
			incorrect = self._wrong.get(name,0)
			total_incorrect += incorrect
			print("class = {0:{sz}} => correct = {1}, incorrect = {2}, ratio = {3:.4f}".format(name,correct,incorrect, correct/(correct + incorrect),sz=max_name_size))
		print("total => correct = {0}, incorrect = {1}, ratio = {2:.4f}".format(total_correct, total_incorrect, total_correct/(total_correct+total_incorrect)))
		print()

	def _classify(self, attributes):
		results = []
		n = len(attributes)
		for k in range(len(self._classes)):
			domain = self._attributes_test[-self._nclasses+k][1]
			max_ll = -sys.maxsize
			clss = None
			for c in domain:
				S = []
				for i in range(n):
					a_i = attributes[i]
					if self._N.get(( (i,a_i), (k,c) )) > self._threshold:
						ll = (1-n)*math.log(self._N.get(( (i,a_i), (k,c) ), len(domain)))
						for j in range(n):
							if j == i:
								continue
							a_j = attributes[j]
							ll += math.log(self._N.get(( (j,a_j), (i,a_i), (k,c) ), 1))
						S.append(ll)
				avg_ll = -sys.maxsize
				if len(S) > 0:
					avg_ll = sum(S)/len(S)
				if avg_ll > max_ll:
					clss = c
					max_ll = avg_ll
			results.append(clss)
		return results

	def _remove_comments(self, line):
		comment = line.find("%")
		if comment > 0:
			line = line[:comment]
		return line
