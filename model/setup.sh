sudo apt remove python3-blinker -y
pip install -r /root/model/requirements.txt
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
huggingface-cli download estieeee/yelp2018_processed --local-dir=/root/data/ --repo-type=dataset