mkdir train_set
cd train_set
wget https://gist.githubusercontent.com/tejus-gupta/3d4564e624cad79691706a5c1303f4c6/raw/3cafe4877f981e3f3c481727d0a90db519a4e95b/download.py
python download.py
unzip masks.zip
unzip train_data.zip
cd ..
git clone https://github.com/tejus-gupta/Segmentation
cd Segmentation
git checkout modelD
