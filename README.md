# PIViT
This is the official pytorch implementation of "PIViT: Large Deformation Image Registration with Pyramid-Iterative Vision Transformer" (MICCAI 2023), written by Tai Ma, Xinru Dai, Suwei Zhang and Ying Wen. Paper link: https://link.springer.com/chapter/10.1007/978-3-031-43999-5_57
![image](https://github.com/Torbjorn1997/PIViT/blob/main/fig2.png)
## Environment
We reimplemented the code on pytorch 1.13 and python 3.7.15. 
## Dataset
We performed retraining，validation and testing on the Mindboggle dataset. 

We provide pre-trained models on the Mindboggle dataset, trained with two subsets, NKI-RS and NKI-TRT, with images cropped to the size of (160, 192, 160).

## Citation
If you use the code in your research, please cite:
```bibtex
@inproceedings{ma2023pivit,
  title={PIViT: Large Deformation Image Registration with Pyramid-Iterative Vision Transformer},
  author={Ma, Tai and Dai, Xinru and Zhang, Suwei and Wen, Ying},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={602--612},
  year={2023},
  organization={Springer}
}
```
The overall framework of the code and the Swin Transformer module are based on [VoxelMorph](https://github.com/voxelmorph/voxelmorph) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration), whose contributions are greatly appreciated.
## Test
We provide the [pre-trained model](https://drive.google.com/file/d/1yjFr6bcjv-UcSXVA-VI2UkwWwfpNSTqu/view?usp=drive_link) and two images for testing from the MMRR subset of the Mindboggle dataset. You can test it with the following code:
```code
python test.py --scansdir   data/vol --labelsdir  data/seg --dataset mind --labels  data/label_mind.npz --model model/0980.pt --gpu 0
```
The test results are:
![image](https://github.com/Torbjorn1997/PIViT/assets/28394656/72748cef-569d-49cc-ad00-1474164191c2)
