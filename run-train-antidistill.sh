for KDCOEF in 0.0001 0.00003 0.00001;do
  bash train-antidistill.sh $KDCOEF
done