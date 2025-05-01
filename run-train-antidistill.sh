for KDCOEF in 0.00003 0.00001;do
  bash train-antidistill.sh $KDCOEF 512
done