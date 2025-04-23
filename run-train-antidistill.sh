for KDCOEF in 1 0.1 0.01 0.001;do
  bash train-antidistill.sh $KDCOEF
done