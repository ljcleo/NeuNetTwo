# NeuNetTwo --- Part I

Neural Network &amp; Deep Learning Final Project (Part I)

## Inference on your own data-set

To run inference, you should first put the model in the the directory `checkpoints/`, then put the images that you want into the directory `data/leftImg8bit`. 

After doing the above things, you can run the following commands :

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/this/dir"
python -m torch.distributed.launch --nproc_per_node=1 experiment/validation.py --output_dir "/path/to/output/dir" --model_path checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth --arch "ocrnet.HRNet_Mscale" --hrnet_base "48" --has_edge True
```
Note that the output images are the classifications, we need to convert the class to the corresponding color. You can run the `experiment/color.py` (change the `img_dir` to the output direction) to do this. If you want to synthesize video from pictures, you should run the `experiment/make_vedio.py`.



## Acknowledgements:

This repository shares code with the following repositories:

* InverseForm: [https://github.com/NVIDIA/semantic-segmentation](https://github.com/Qualcomm-AI-research/InverseForm)

We would like to acknowledge the researchers who made these repositories open-source.


