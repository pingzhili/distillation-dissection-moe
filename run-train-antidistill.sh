for KDCOEF in 0.01 0.001 0.0001 0.00001;do
  bash train-antidistill.sh $KDCOEF
done