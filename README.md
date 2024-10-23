# DSIN-PMA
### Dual-Stream Interactive Networks with Pearson-Mask Awareness for Multivariate Time Series Forecasting

## Overall Architecture
Overall structure of DSIN-PMA, consisting of dual-stream network. 1) Dual-stream embedding layer: Multivariate embedding and time-step embedding effectively represent the multivariate time series data from two dimensions. 2) Dual-stream encoder layer: i) Cross-multivariate attention with Pearson-mask encoder introduces a Pearson mask knowledge matrix into the traditional attention mechanism, aimed at de-noising and capturing cross-variate dependencies. ii) Time-step attention encoder focuses on learning the seasonality and trend within the time steps, using a conventional multi-head attention mechanism to capture seasonal information. 3) Cross-dimensional consistent learning: A consistency loss function is applied to encourage the dual-stream encoder to produce similar outputs, enhancing the model's generalization ability.  
![Figure3](https://github.com/user-attachments/assets/c70febbd-b5aa-416e-a8a6-0251e9966dfc)


## Main Result of Multivariate Forecasting

## Efficiency on Long Look-back Windows
Our DSIN-PMA consistently reduces the MSE scores as the look-back window increases, which confirms our model’s capability to learn from longer receptive field.
![Figure8](https://github.com/user-attachments/assets/9ea86b9c-4559-4e59-b9a6-d25bd8b8aba0)

## Getting Started
1. Install requirements.  
2. Download data. You can download all the datasets from Google Drive. Create a seperate folder ./dataset and put all the csv files in the directory.
3. Training. All the scripts are in the directory ./scripts/multivariate_forecast. For example, if you want to get the multivariate forecasting results for ETTh1 dataset, just run the following command, and you can open ./result.txt to see the results once the training is done:  

    sh ./scripts/multivariate_forecast/ETTh1.sh  
   
You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths.). We also provide codes for the baseline models.

## Acknowledgement
We appreciate the following github repo very much for the valuable code base and datasets:  
https://github.com/cure-lab/LTSF-Linear  
https://github.com/thuml/Autoformer  
https://github.com/alipay/Pyraformer  
https://github.com/zhouhaoyi/Informer2020  
https://github.com/ts-kim/RevIN  
https://github.com/timeseriesAI/tsai  
https://github.com/MAZiqing/FEDformer  
https://github.com/yuqinie98/patchtst  

## Contact
If you have any questions or concerns, please contact us: yejunjie@stu.yun.edu.cn or zhaochunna@ynu.edu.cn or submit an issue.
