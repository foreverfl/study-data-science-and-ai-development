import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 변수 중요도 plot1


def plot_feature_importance(importance, names, topn='all'):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names,
            'feature_importance': feature_importance}
    fi_temp = pd.DataFrame(data)

    fi_temp.sort_values(by=['feature_importance'],
                        ascending=False, inplace=True)
    fi_temp.reset_index(drop=True, inplace=True)

    if topn == 'all':
        fi_df = fi_temp.copy()
    else:
        fi_df = fi_temp.iloc[:topn]

    plt.figure(figsize=(10, 8))
    sns.barplot(x='feature_importance', y='feature_names', data=fi_df)

    plt.xlabel('importance')
    plt.ylabel('feature names')
    plt.grid()

    return fi_df

# 변수 중요도 plot2


def plot_PFI(pfi, col_names):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    for i, vars in enumerate(col_names):
        sns.kdeplot(pfi.importances[i], label=vars)
    plt.legend()
    plt.grid()

    sorted_idx = pfi.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(pfi.importances[sorted_idx].T,
                vert=False, labels=col_names[sorted_idx])
    plt.axvline(0, color='r')
    plt.grid()
    plt.show()


# 데이터 로딩
current_path = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 경로 가져오기
parent_path = os.path.dirname(current_path)
csv_path = os.path.join(
    parent_path, 'data', 'smoking_driking_dataset_Ver01.csv')
data = pd.read_csv(csv_path)

categorical_features = ['sex', 'SMK_stat_type_cd']
numerical_features = ['age', 'height', 'weight', 'waistline', 'sight_left', 'sight_right', 'hear_left', 'hear_right',
                      'SBP', 'DBP', 'BLDS', 'tot_chole', 'HDL_chole', 'LDL_chole', 'triglyceride', 'hemoglobin',
                      'urine_protein', 'serum_creatinine', 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP']

data = pd.get_dummies(data[['sex', 'SMK_stat_type_cd']], prefix=[
                      'sex', 'SMK_stat_type_cd'])
print(data.head())
print(data.info())


# data: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset
