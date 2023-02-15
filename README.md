# Paper in Submission

In this work, we propose a novel image representation method called CDCR, which combines discrete representation (DR) and continuous representation (CR) of images. CDCR can serve as a plug-in module to help image super-resolution frameworks accommodate arbitrary scale factors. 

We have applied our method to various remote sensing datasets, such as RSC11, RSSCN7 and WHU-RS19.

![CDCR](https://github.com/Suanmd/CDCR/blob/main/img/example.png)


### Training commands

    python train.py --config configs/train/train_edsr-baseline-cdcr.yaml --gpu 0
    python train.py --config configs/train/train_rdn-cdcr.yaml --gpu 0
    python train.py --config configs/train/train_rcan-cdcr.yaml --gpu 0

### Testing commands

    bash scripts/testSCA-RSC11.sh save/model_name/epoch_num.pth gpu_num ABCDEFGHIJK
    bash scripts/testSCA-RSSCN7.sh save/model_name/epoch_num.pth gpu_num ABCDEFGHIJK
    bash scripts/testSCA-WHU-RS19.sh save/model_name/epoch_num.pth gpu_num ABCDEFGHIJK

Note that `A`~`K` represent scale factors of 2, 3, 4, 6, 8, 12, 16, 20, 3.4, 9.7, and 17.6, respectively.

### References

 1. [LIIF](https://github.com/yinboc/liif)
 2. [Meta-SR](https://github.com/XuecaiHu/Meta-SR-Pytorch)
 3. [A-LIIF](https://github.com/LeeHW-THU/A-LIIF)
 4. [ArbSR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR)

