# covid_emr

## Model Zoo

### Machine Learning Models

- [ ] Logistic regression (LR)
- [ ] Random forest (RF)
- [ ] Decision tree (DT)
- [ ] Gradient Boosting Decision Tree (GBDT)
- [ ] XGBoost

### Deep Learning Models

- [ ] Multi-layer perceptron (MLP)
- [ ] Recurrent neural network (RNN)
- [ ] Long-short term memory network (LSTM)
- [ ] Gated recurrent units (GRU)
- [ ] Temporal convolutional networks
- [ ] Transformer

### EHR Predictive Models

- [ ] RETAIN
- [ ] StageNet
- [ ] Dr. Agent
- [ ] AdaCare
- [ ] ConCare
- [ ] GRASP
- [ ] (CovidCare)

#### AdaCare

```python
model = models.AdaCare()
"""
default AdaCare hyperparameter:

hidden_dim=128, kernel_size=2, kernel_num=64, input_dim=76, output_dim=1, dropout=0.5, r_v=4, r_c=4, activation='sigmoid', device='cuda')

In mimic-iii dataset, the input shape is T=48, H=76
"""
x = torch.randn(batch_size, 48, 76)
output, inputse_att = model(x, device)
```

TODO: 不需要attention matrix的输出

## 各数据上，特征的记录频率统计

> 区分出高频、低频特征

- [ ] 根据missing rate去筛选特征，此处missing rate计算时应是patient level, not visit level
- [ ] 检查一下value非numeric的特征的记录情况

## Evaluate and Loss Calculation

- [ ] 评估时按每一次visit对应的预测结果 (mortality outcome / los) 的结果进行评估
- [ ] 计算 loss 时才取平均，评估时将patient每一次visit时的结果取平均没有意义

## Bugs

- [ ] 检查 predict LOS 时为何不收敛

## Model Design

- [ ] Multi-task learning: 2 branches: mortality and LOS prediction (2 tasks)
