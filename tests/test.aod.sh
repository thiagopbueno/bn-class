echo
echo '=== DATASET: emotions ==='
../src/bnclass.py aod ../datasets/emotions-train.arff ../datasets/emotions-test.arff --classes 6  --threshold 30 -v 1 

echo
echo '=== DATASET: yeast ==='
../src/bnclass.py aod ../datasets/yeast-train.arff    ../datasets/yeast-test.arff    --classes 14 --threshold 30 -v 1

echo
echo '=== DATASET: yelp ==='
../src/bnclass.py aod ../datasets/yelp-train.arff     ../datasets/yelp-test.arff     --classes 8  --threshold 30 -v 1

echo
echo '=== DATASET: medical ==='
../src/bnclass.py aod ../datasets/medical-train.arff  ../datasets/medical-test.arff  --classes 45 --threshold 30 -v 1