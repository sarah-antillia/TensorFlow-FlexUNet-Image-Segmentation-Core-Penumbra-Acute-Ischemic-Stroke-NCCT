<h2>TensorFlow-FlexUNet-Image-Segmentation-Core-Penumbra-Acute-Ischemic-Stroke-NCCT (2025/09/25)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for CPAISD (Core-Penumbra Acute Ischemic Stroke Dataset) Non Contrast Cranial CT, based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
 and a 512x512 pixels 
<a href="https://drive.google.com/file/d/14MtRxPujs3vRBh7M2VsV7aZj9-MsZNRg/view?usp=sharing">CPAISD-PNG-ImageMask-Subset.zip</a>
with colorized masks (Core:red, Penumbra:green), which was derived by us from <b>train</b> subset of 
<a href="https://zenodo.org/records/10892316">
<b>CPAISD: Core-Penumbra Acute Ischemic Stroke Dataset</b></a>
<br>
<br>
On singleclass segmentation for Ischaemic-Stroke-NCCT, please refer to our expriment 
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Automated-Ischaemic-Stroke-NCCT">TensorFlow-FlexUNet-Image-Segmentation-Automated-Ischaemic-Stroke-NCCT</a>
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
our dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<b>rgb_map (Core:red, Penumbra:green)</b><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/images/10026.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/masks/10026.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test_output/10026.png" width="320" height="320"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/images/10124.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/masks/10124.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test_output/10124.png" width="320" height="320"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/images/10135.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/masks/10135.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test_output/10135.png" width="320" height="320"></td>
</tr>
</table>

<hr>
<br>

<h3>1. Dataset Citation</h3>
The dataset used here was obtained from 
<br><br>
<a href="https://zenodo.org/records/10892316">
<b>CPAISD: Core-Penumbra Acute Ischemic Stroke Dataset</b></a>
<br>
<br>
<b>About Dataset</b><br>
The Core-Penumbra Acute Ischemic Stroke Dataset (CPAISD) provides 112 
anonymized CT scans from hyperacute stroke patients. <br><br>
<b>Experts have manually delineated the ischemic core and penumbra on every relevant slice.</b> <br>
Anonymized with Kitware DicomAnonymizer, it retains key DICOM fields for demographic and domain shift studies:
<br>
<br>
The dataset is split into three folds for robust research:
<br>
Training: 92 studies, 8,376 slices<br>
Validation: 10 studies, 980 slices<br>
Testing: 10 studies, 809 slices <br>
<br>
DOI: <a href="https://doi.org/10.5281/zenodo.10892316">https://doi.org/10.5281/zenodo.10892316</a>
<br><br>


<b>Citation:</b><br>
Umerenkov, D., Kudin, S., Peksheva, M., & Pavlov, D. (2024). CPAISD: Core-Penumbra Acute Ischemic Stroke Dataset [Data set].<br>
 Zenodo. https://doi.org/10.5281/zenodo.10892316 <br>
<br>

<b>License:</b><br> 
<a href="https://creativecommons.org/licenses/by/4.0/legalcode">
Creative Commons Attribution 4.0 International</a>
<br>
<br>
<h3>
<a id="2">
2 CPAISD ImageMask Dataset
</a>
</h3>
<h4>2.1 Download CPAISD-PNG-ImageMask-Subset</h4>
 If you would like to train this CPAISD Segmentation model by yourself,
 please download  our dataset <a href="https://drive.google.com/file/d/14MtRxPujs3vRBh7M2VsV7aZj9-MsZNRg/view?usp=sharing">CPAISD-PNG-ImageMask-Subset.zip</a> on the google drive
, expand the downloaded and put it under <b>./dataset</b> folder to be.<br>
<pre>
./dataset
└─CPAISD
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>CPAISD Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/CPAISD/CPAISD_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large to use for a training set of our segmentation model.
<br>
<br>
<h4>2.2 PNG Subset Generation </h4>
For simplity and reducing a training time, we generated our dataset with colorized masks from the <i>image.npz</i> and <i>mask.npz</i> 
files in <b>train</b> subset only in the following original CPAISD dataset.
<br>
<br>
<b>CPAISD Dataset Structure</b><br>
<pre>
./dataset
├─test
│  ├─2.25.116613827163187469445304378258728959191
│  │  ├─00000
             ├─image.npz
             └─mask.npz
...
├─train
│  ├─2.25.10536893979831470310119796877031918393
│  │  ├─00000
             ├─image.npz
             └─mask.npz
...
└─val
    ├─2.25.127914288592697535538245879784450329006
    │  ├─00000
             ├─image.npz
             └─mask.npz
</pre>
<b>2.3 Exclusion and Colorization of Masks </b><br>
<ul>
<li>We excluded all black empty masks and their corresponding images,
 which were irrelevant for training our segmentation model, from the original dataset,</li>
<li>We generated colorized masks (Core:red, Penumbra:green) from the <i>mask.npz</i> files.
</li>
</ul>
This time, we used the original <i>image.npz</i> files to generate the dataset without any enhancement. 
However, you can also try to generate normalized images from them, which are more recognizable to the human eye as shown below.
This will cause some pixel-level changes in the original images, which might be harmful to build a better segmentation model.<br>
<table>
<tr>
<th>Original image</th>
<th>Normalized image</th>
<th>Ground Truth</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/asset/non-normalized-10022.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/asset/normalized-10022.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/asset/mask-10022.png" width="320" height="320"></td>

<tr>
</table>

<br>
<h4>2.4 PNG Dataset Images and Masks </h4>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorFlowUNet Model
</h3>
 We have trained CPAISD TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/CPAISD/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/CPAISDand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small base_filters=16 and large base_kernels=(9,9) for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 3

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learning_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for CPAISD 1+2 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"

;CPAISD rgb color map dict for 3 classes.
;   Background:black, Penumbra:green, Stroke Core:red
rgb_map = {(0,0,0):0,(0,255,0):1, (255,0,0):2}

</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 34,35 36)</b><br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 70,71,72)</b><br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 72 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/asset/train_console_output_at_epoch72.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/CPAISD/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/CPAISD/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/CPAISD</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for CPAISD.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/asset/evaluate_console_output_at_epoch72.png" width="720" height="auto">
<br><br>Image-Segmentation-CPAISD

<a href="./projects/TensorFlowFlexUNet/CPAISD/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this CPAISD/test was not so low, but dice_coef_multiclass high as shown below.
<br>
<pre>
categorical_crossentropy,0.0473
dice_coef_multiclass,0.9757
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/CPAISD</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for CPAISD.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/CPAISD/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
As shown below, this segmentation model failed to detect some Ischemia lesions.<br>

<img src="./projects/TensorFlowFlexUNet/CPAISD/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/images/10030.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/masks/10030.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test_output/10030.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/images/10124.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/masks/10124.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test_output/10124.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/images/10337.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/masks/10337.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test_output/10337.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/images/10646.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/masks/10646.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test_output/10646.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/images/11263.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/masks/11263.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test_output/11263.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/images/11871.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test/masks/11871.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CPAISD/mini_test_output/11871.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Automated Segmentation of Ischemic Stroke Lesions in Non-Contrast Computed Tomography Images for Enhanced Treatment and Prognosis</b><br>
Toufiq Musah, Prince Ebenezer Adjei, Kojo Obed Otoo<br>

<a href="https://arxiv.org/html/2411.09402">
https://arxiv.org/html/2411.09402
</a>
<br>
<br>
<b>2. APIS: a paired CT-MRI dataset for ischemic stroke segmentation - methods and challenges </b><br>
Santiago Gómez, Edgar Rangel, Daniel Mantilla, Andrés Ortiz, Paul Camacho, Ezequiel de la Rosa, Joaquin Seia, <br>
Jan S. Kirschke, Yihao Li, Mostafa El Habib Daho & Fabio Martínez<br>
<a href="https://www.nature.com/articles/s41598-024-71273-x">
https://www.nature.com/articles/s41598-024-71273-x
</a>
<br>
<br>
<b>3.Core-Penumbra Hyperacute Ischemic Stroke Dataset </b><br>
D. Umerenkov, S. Kudin, M. Peksheva & D. Pavlov<br>
<a href="https://www.nature.com/articles/s41597-025-05000-0">
https://www.nature.com/articles/s41597-025-05000-0</a>
<br>
<br>
<b>4.Segmenting Ischemic Penumbra and Infarct Core Simultaneously on <br>
Non-Contrast CT of Patients with Acute Ischemic Stroke Using Novel Convolutional Neural Network</b><br>
Hulin Kuang, Xianzhen Tan, Jie Wang, Zhe Qu, Yuxin Ca, Qiong Chen, Beom Joon Kim, and Wu Qiu<br>
<a href="https://www.mdpi.com/2227-9059/12/3/580">
https://www.mdpi.com/2227-9059/12/3/580
</a>
<br>
<a href=" https://doi.org/10.3390/biomedicines12030580">https://doi.org/10.3390/biomedicines12030580</a>
<br>
<br>
<b>5.TensorFlow-FlexUNet-Image-Segmentation-Automated-Ischaemic-Stroke-NCCT</b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Automated-Ischaemic-Stroke-NCCT">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Automated-Ischaemic-Stroke-NCCT</a>

