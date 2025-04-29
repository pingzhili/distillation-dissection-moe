## Setup

```bash
bash setup.sh
```

## Run Experiments

### Exp 1: Coef=0.00001
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train-distill.sh "qwen-antidistill-coef0.00001-temp2-epoch2-lr5e-5-checkpoint-60"
```

### Exp 2: Coef=0.00003
```bash
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 bash train-distill.sh "qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60"
```