### NAFNet-Dual

------

Requirement:

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3

git clone https://github.com/RicoJYang/NAFNet-Dual.git
cd NAFNet-Dual
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

You can get pretrained model from the releases part of this github repo:

https://github.com/RicoJYang/NAFNet-Dual/releases/download/pretrained_model/team14_RTAFNet.pth

or use shell :

```
wget https://github.com/RicoJYang/NAFNet-Dual/releases/download/pretrained_model/team14_RTAFNet.pth 
```

This pretrained model use option file which located in './options/test/All/NAFNet-2Phase-384midRes-test.yml'

You can change the pretrain model path[team14_RTAFNet.pth ] in this option file to test  pretrain model and run the script under:

```shell
python basicsr/demo.py -opt ./options/test/All/NAFNet-2Phase-384midRes-test.yml --input_path [noisy image path] --output_path [output path]
```

#### How to run demo:

```shell
python basicsr/demo.py -opt [test option file path] --input_path [noise image path] --output_path [out put path]
```

Example:

`````shell
python basicsr/demo.py -opt /mnt/lustre/GPU7/home/yangbo/workspace/codes/NAFNet-raw/options/test/All/NAFNet-2Phase-384midRes-test.yml --input_path /mnt/lustre/GPU7/home/yangbo/workspace/data/FinalTest/NoisingImg/ --output_path /mnt/lustre/GPU7/home/yangbo/workspace/data/FinalTest/NAFNet_384_midresidue_RESULT/
`````

#### How to test dataset:

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt [test option file path]  --launcher pytorch
```

Example:

```shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt /mnt/lustre/GPU7/home/yangbo/workspace/codes/NAFNet-raw/options/test/All/NAFNet-2Phase-384midRes-test.yml  --launcher pytorch
```

#### How to train :

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt [train option file path] --launcher pytorch
```

Example:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt /mnt/lustre/GPU7/home/yangbo/workspace/codes/NAFNet-raw/options/train/All/NAFNet-ALL-2Phase-384-midRes.yml --launcher pytorch
```

------

Before test or train,you should edit the option file to make sure input correct gt&lq path and pretrain model.