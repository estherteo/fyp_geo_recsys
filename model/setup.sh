sudo apt remove python3-blinker -y
pip install -r ./requirements.txt
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
huggingface-cli download estieeee/yelp2018_processed --local-dir=./data/ --repo-type=dataset
mkdir ./dataset_challenge
cp ./data/* ./dataset_challenge/
