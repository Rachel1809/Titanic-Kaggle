import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def Preprocessing(test_data):
    train_data = pd.read_csv('train.csv', index_col="PassengerId")

    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    scaler = StandardScaler()

    def one_hot_encoding(data, cols, isTest=False):
        if (isTest):
            result = ohe.transform(pd.DataFrame(data[cols]))
        else:
            result = ohe.fit_transform(pd.DataFrame(data[cols]))
        OH_col = pd.DataFrame(result, columns=ohe.get_feature_names_out(), index=data.index)
        remain_data = data.drop(cols, axis=1)
        data = pd.concat([remain_data, OH_col], axis=1)
        return data

    def impute_data(data, cols, strategy):
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        data[cols] = imputer.fit_transform(data[cols])
        return data

    def standard_scaler(data, cols, isTest=False):
        if (isTest):
            data[cols] = scaler.transform(data[cols])
        else:
            data[cols] = scaler.fit_transform(data[cols])
        return data

    def clean(data, isTest=False):
        data['Cabin'] = data['Cabin'].astype('str').str[0]
        data = data.drop(["Ticket", "Name"], axis=1)

        cate_cols = ["Pclass", "SibSp", "Parch"]
        for col in cate_cols:
            data[col] = data[col].astype('int64')

        num_cols = ["Fare", "Age"]
        label_cols = ["Sex", "Embarked", "Cabin"]

        data = one_hot_encoding(data, label_cols, isTest)
        data = impute_data(data, num_cols, 'mean')
        data = impute_data(data, cate_cols, 'median')
        data = standard_scaler(data, num_cols, isTest)

        return data


    clean(train_data, isTest=False)
    test_df = clean(test_data, isTest=True)
    
    return test_df