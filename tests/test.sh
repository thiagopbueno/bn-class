echo '=== DATASET: emotions ==='
../src/bnclass.py nbc ../datasets/emotions-train.arff ../datasets/emotions-test.arff --classes 6  -v 1
echo '=== DATASET: yeast ==='
../src/bnclass.py nbc ../datasets/yeast-train.arff    ../datasets/yeast-test.arff    --classes 14 -v 1
echo '=== DATASET: yelp ==='
../src/bnclass.py nbc ../datasets/yelp-train.arff     ../datasets/yelp-test.arff     --classes 8  -v 1
echo '=== DATASET: medical ==='
../src/bnclass.py nbc ../datasets/medical-train.arff  ../datasets/medical-test.arff  --classes 45 -v 1