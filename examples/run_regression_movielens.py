import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
import numpy as np

if __name__ == "__main__":
    print(1111111)

    data = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip"]
    target = ['rating']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        # 将sparsefeature转成连续的ID
        data[feat] = lbe.fit_transform(data[feat])
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    # print("train_model_input:", train_model_input.shape())
    test_model_input = {name: test[name] for name in feature_names}
    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
    model.compile("adam", "mse", metrics=['mse'], )
    print("train_model_input:", len(train_model_input), len(train_model_input.keys()), len(train_model_input.values()))
    for key in train_model_input.keys():
        print(key, train_model_input[key], type(train_model_input[key]) )
    x = train_model_input
    if isinstance(x, dict):
        x = [x[feature] for feature in x.keys()]
    print("===========================")
    print(type(x))
    print(type(x[0]))
    
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)
    print('before x=========')
    print(type(x))
    print(len(x))
    print('before x[0]=========')
    print(type(x[0]))
    print(x[0].shape)
    # print(x[0])
    
    print("after=============")
    x = np.concatenate(x, axis=-1)
    print('after x==========')
    print(type(x))
    print(x.shape)

    print('after x[0]==========')
    print(type(x[0]))
    print(x[0].shape)
    print(x[0])
    

    history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test MSE", round(mean_squared_error(
        test[target].values, pred_ans), 4))
