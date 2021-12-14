# Mobile-PolypNet
## Introduction:
Colorectal cancer (CRC) is the third leading type of cancer globally, and the second principal cause of cancer-related death in the United States. Approximately 4% of the female and 4.3% of the male population in the United States suffer from colorectal cancer. However, with early detection and proper treatment, 90% of the patients have an increased life span of more than five years.

Over the years, different traditional image processing and deep learning networks have been proposed. Although deep learning models outperformed classical image processing, they require high computational resources. A metric to evaluate the computational complexity of deep learning networks is  frames-per-second (FPS) to evaluate the its processing rate. However, this is a platform dependent metric. A more common, platform independent, metric  is number of floating-point operations (FLOPs) that network executes in order to achieve the task.

In this paper, we proposed a novel lightweight image segmentation architecture that is significantly less complex, requiring a fraction of training parameters and a lower number of FLOPs. This significant reduction in complexity is achieved while maintaining high accuracy. Our model achieved state-of-the-art performance in the test dataset. The significance of this works is in its novel encoder-decoder architecture backbone that is lightweight, suitable for mobile deployment devices. Based on the empirical evidence, we adopted  Dice coefficient as objective loss function, which yields more accurate results. We used the same training and testing sets as PraNet \cite{fan2020pranet} and performed extensive testing using important semantic segmentation metrics for better benchmarking.


## Model:
![alt text](https://github.com/rkarmaka/Mobile-PolypNet/figs/model_arch_mod.png?raw=true)

## Results:
![alt text](https://github.com/rkarmaka/Mobile-PolypNet/figs/out_2.png?raw=true)
