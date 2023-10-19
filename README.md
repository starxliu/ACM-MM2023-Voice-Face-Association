## Taking a Part for the Whole: An Archetype-agnostic Framework for Voice-Face Association

Implementing a portion of the paper's code using the PyTorch framework.



### Environment

- Ubutu 20.04
- Python 3.8.12
- Pytorch 1.4.0 
- CUDA 10.1

### Requirements

```txt
easydict==1.9
librosa==0.8.0
lmdb==0.94
matplotlib==3.5.0
numpy==1.20.3
opencv_python==4.5.4.60
PyYAML==6.0
Pillow==8.4.0
scipy==1.7.3
scikit_learn==1.0.1
tensorboardX==2.1
torch==1.4.0
torchaudio==0.4.0
torchvision==0.5.0
tqdm==4.62.3
```



### Preliminary

1. Get dataset

   - Download the [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html), [VGGFace](https://www.dropbox.com/s/bqsimq20jcjz1z9/VGG_ALL_FRONTAL.zip?dl=0) datasets and unzip them to the path specified in the `image_data_dir/audio_data_dir` section of the `cfg.yaml` file.

2. Get our trained model

   - Download the trained [model](https://www.dropbox.com/s/kllvfxyoq0bjfcb/checkpoint_best.pth.tar?dl=0). You can configure the model path in `evaluation.trained_model`of the `cfg.yaml` file

3.  Get test list

      - The test list contains three evaluation scenarios : matching, verification and retrieval.

      - Download [test list](https://www.dropbox.com/s/ht5g2hjzjs2q0hb/gen_list.zip?dl=0) or generate your own test list. You can also customize the path to the test list by modifying the `list_dir` key in the configuration file.

### Training

Training strategies are coming soon.

### Evaluation on our trained model

```shell
python3 eval.py --cfg config/cfg.yaml
```

