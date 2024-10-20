# DSIN-PMA
## Dual-Stream Interactive Networks with Pearson-Mask Awareness for Multivariate Time Series Forecasting

## Overall Architecture

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
