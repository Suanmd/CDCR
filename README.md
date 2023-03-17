# Scale-arbitrary Image Super-resolution Task
## Paper in Submission

After surveying the literature on scale-arbitrary super-resolution, we summarize existing methods into two categories: one-stage and two-stage. The two-stage method involves learning the discrete representation (DR) of an image, requiring the prediction of convolution kernels, while the one-stage method involves learning the continuous representation (CR) of an image, directly mapping coordinates to pixel values.

This study introduces a novel image representation method, named CDCR, which integrates the DR and CR of an image. CDCR is designed as a plug-in module to enable existing super-resolution frameworks to handle arbitrary scale factors. By integrating the advantages of DR and CR, CDCR enhances the overall image accuracy. The CR-based dense prediction module provides a strong initial estimation of the high-resolution image, while the DR-based resolution-specific refinement module fine-tunes the predicted values of local pixels to achieve high accuracy.

We apply our method to various remote sensing datasets, such as RSC11, RSSCN7, and WHU-RS19, achieving advanced performance in this field. Our approach is based on the work of [LIIF](https://github.com/yinboc/liif), but with modifications specific to remote sensing scenarios.

Overall, the proposed CDCR method advances the development of more advanced scale-arbitrary image super-resolution techniques.

![CDCR](https://github.com/Suanmd/CDCR/blob/main/img/example.png)

### Training commands

Train the model directly by running `train.py`.

    python train.py --config configs/train/train_edsr-baseline-cdcr.yaml --gpu 0
    python train.py --config configs/train/train_rdn-cdcr.yaml --gpu 0
    python train.py --config configs/train/train_rcan-cdcr.yaml --gpu 0

### Testing commands
1) The CDCR inference command:

	    bash scripts/test-RSC11.sh save/model_name/epoch_num.pth gpu_num ABCDEFGHIJK
	    bash scripts/test-RSSCN7.sh save/model_name/epoch_num.pth gpu_num ABCDEFGHIJK
	    bash scripts/test-WHU-RS19.sh save/model_name/epoch_num.pth gpu_num ABCDEFGHIJK

2) The CDCR+ inference command:

	    bash scripts/testSCA-RSC11.sh save/model_name/epoch_num.pth gpu_num ABCDEFGHIJK
	    bash scripts/testSCA-RSSCN7.sh save/model_name/epoch_num.pth gpu_num ABCDEFGHIJK
	    bash scripts/testSCA-WHU-RS19.sh save/model_name/epoch_num.pth gpu_num ABCDEFGHIJK

	Note that `A`~`K` represent scale factors of 2, 3, 4, 6, 8, 12, 16, 20, 3.4, 9.7, and 17.6, respectively.

### References

 1. [LIIF](https://github.com/yinboc/liif)
 2. [Meta-SR](https://github.com/XuecaiHu/Meta-SR-Pytorch)
 3. [A-LIIF](https://github.com/LeeHW-THU/A-LIIF)
 4. [ArbSR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR)


