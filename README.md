# CO2Net
Official implementation of Deep Video Harmonization  with Color Mapping Consistency

## Prepare
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Clone this repo:
```bash
git clone https://github.com/bcmi/Image_Harmonization_Datasets.git
cd Image_Harmonization_Datasets
```

### install cuda package
```bash
cd CO2Net
cd trilinear
. ./setup.sh
```

```bash
cd CO2Net
cd tridistribute
. ./setup.sh
```
## Generate frame-level text
```bash
python3 scripts/generate_frame_list.py --dataset_path <Your path to HYouTube>
```
## evaluate by our released model
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone ./final_models/issam_backbone.pth --previous_num 8 --future_num 8  --use_feature --checkpoint ./final_models/issam_final.pth
```
Or evaluate without refinement module

```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone ./final_models/issam_backbone.pth --previous_num 8 --future_num 8 
```
Your can also use your own backbone or whole models.

## Train your own model
Your can directly train by 
```bash
python3  scripts/my_train.py --gpu=1 --dataset_path <Your path to HYouTube> --train_list ./train_list.txt --val_list ./test_frames.txt --backbone <Your backbone model> --backbone_type <Your backbone type, we provide 'issam' and 'rain' here> --previous_num 8 --future_num 8 --use_feature --normalize_inside --exp_name <exp name>
```
But since we adopt two stage traing strategy, we highly recommand your to calculate and store the result of Lut like 

```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone <Your backbone model> --previous_num 8 --future_num 8 --write_lut_output <directory to store lut output> --write_lut_map <directory to store lut map> 
```
then you can use 

```bash
python3  scripts/my_train.py --gpu=1 --dataset_path  <Your path to HYouTube> --train_list ./train_list.txt --val_list ./test_frames.txt --backbone  <Your backbone model> --previous_num 8 --future_num 8 --use_feature --normalize_inside --exp_name <exp_name> --lut_map_dir <directory to store lut map> --lut_output_dir <directory to store lut output>
```

Then your can evaluate it by above instruction

## Evaluate temporal consistency
we need you to download HYouTube_next from [link] and install Flownet2
### prepare
Please follow command of [FlowNetV2](https://github.com/NVIDIA/flownet2-pytorch) to install and download FlowNetV2 weight.

### prepare result
You need to store the numpy result of model like 
```bash
python3  scripts/evaluate_model.py --gpu=0 --dataset_path <Your path to HYouTube> --val_list ./test_frames.txt --backbone <Your backbone model> --previous_num 8 --future_num 8 --write_npy_result --result_npy_dir <Directory to store numpy result>
```


