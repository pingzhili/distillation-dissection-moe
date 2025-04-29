## Setup

```bash
bash setup.sh
```

## Run Experiments

### Exp 1: Coef=0.00001
```bash
bash train-distill.sh "qwen-antidistill-coef0.00001-temp2-epoch2-lr5e-5-checkpoint-60"
```

### Exp 2: Coef=0.00003
```bash
bash train-distill.sh "qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60"
```