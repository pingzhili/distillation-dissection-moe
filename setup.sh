conda create -y -n antidistill python=3.10
conda activate antidistill

mkdir -p data
huggingface-cli download Phando/antidistill-exps --local-dir data/antidistill-exps
mkdir -p outputs
pip install -r requirements.txt

