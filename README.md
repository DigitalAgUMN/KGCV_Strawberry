# KGCV_Strawberry
A knowledge-guided computer vision framework for strawberry fruit detection and growth modeling. 
For detailed methodology please refer to "Qi Yang, Licheng Liu, Junxiong Zhou, Mary Rogers, Zhenong Jin, 2024. Predicting the growth trajectory and yield of greenhouse strawberries based on knowledge-guided computer vision, Computers and Electronics in Agriculture, 220, 108911. https://doi.org/10.1016/j.compag.2024.108911"
# Dataset
The complete dataset collected in this study can be downloaded from https://doi.org/10.5281/zenodo.10957605
# Training 
![fig2_v4](https://github.com/DigitalAgUMN/KGCV_Strawberry/assets/23078349/92e5ef04-e4af-40c9-bbf6-6f8fd2046ce9)
## Train detection network
"KGCV_Strawberry_Train_FasterRCNN.py" is used to train a faster-RCNN for strawberry bounding box and main phenological stage detection
## Train trait estimation network
"KGCV_Strawberry_Train_CNN.py" is for traing a CNN to estimate the fruit size and decimal phenological stage. 
## Train parameter network
"KGCV_Strawberry_Train_MLP-1_parameterNet.py" generates the synthetic data pair first,
```
# load dataset    
syn = loadSyntheticData(mode=mode,ensembleN=1000)
```
where the input is a sparse 1×70 diameter/length sequence, and output is three parameters in the S-shape logistic model
$$f(x) = \frac{y_0 - s}{1 + e^{-r(x-t_0)}} + s$$
where $y_0$, $s$, and $r$ are three curve parameters control the trajectory of fruit size.
The generated synthetic dataset was used to train a MLP to map the 1×70 vector to curve parameters.
# Train fruit growth netwrk
"KGCV_Strawberry_Train_MLP-1_growthNet.py" trains a MLP to map fruit size at $t$ to $t+\Delta t$
$$s_{t+\Delta t} = g(s_t, \Delta GDD, Para)$$
