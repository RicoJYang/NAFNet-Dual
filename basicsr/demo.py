# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import os, sys
from tqdm import tqdm
# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite

# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()
    ## 2. run inference
    opt['dist'] = False
    model = create_model(opt)
    img_path = opt['img_path'].get('input_img')
    output_path = opt['img_path'].get('output_img')

    for imgname in tqdm(os.listdir(img_path)):
        ## 1. read image
        file_client = FileClient('disk')
        print(img_path + imgname)
        img_in = img_path + imgname
        img_bytes = file_client.get(img_in, None)
        try:
            img = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("path {} not working".format(img_path))

        img = img2tensor(img, bgr2rgb=True, float32=True)



        model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imgout = output_path + imgname
        imwrite(sr_img, imgout)

        print(f'inference {img_path} .. finished. saved to {output_path}')

if __name__ == '__main__':
    main()

