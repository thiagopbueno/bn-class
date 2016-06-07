#! /usr/bin/env python3

import time
import argparse

from nbc import NBC
from aod import AOD

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("classifier", choices=["nbc","tan","aod"], help="classifier type")
	parser.add_argument("training", help="path to training dataset")
	parser.add_argument("test", help="path to test dataset")
	parser.add_argument("-c", "--classes", type=int, default=1, help="number of class attributes (multidimensional)")
	parser.add_argument("-t", "--threshold", type=int, default=1, help="threshold for AOD")
	parser.add_argument("-v", "--verbose", type=int, default=0, choices=[0,1,2,3], help="verbose mode (default 0)")
	args = parser.parse_args()

	if args.classifier == "nbc":
		nbc = NBC(args.classes, args.verbose)

		start = time.clock()
		nbc.train(args.training)
		end = time.clock()
		training_uptime = end-start

		start = time.clock()
		nbc.test(args.test)
		end = time.clock()
		testing_uptime = end-start

		print(">> Training time = {0:.3f}sec.".format(training_uptime))
		print(">> Testing time  = {0:.3f}sec.".format(testing_uptime))
	elif args.classifier == "aod":
		aod = AOD(args.classes, args.verbose, args.threshold)

		start = time.clock()
		aod.train(args.training)
		end = time.clock()
		training_uptime = end-start

		start = time.clock()
		aod.test(args.test)
		end = time.clock()
		testing_uptime = end-start

		print(">> Training time = {0:.3f}sec.".format(training_uptime))
		print(">> Testing time  = {0:.3f}sec.".format(testing_uptime))
