import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# PCA 객체 생성 (주성분의 개수를 지정하지 않으면 자동으로 모든 주성분을 유지합니다)
pca = PCA()

# PCA 모델 훈련
pca.fit(train)

# 변환된 데이터 얻기 (주성분으로 데이터를 변환)
transformed_data = pca.transform(train)

# 주성분의 정보 출력
print("주성분 (Principal Components):")
print(pca.components_)

# 주성분의 설명된 분산 출력
print("\n주성분의 설명된 분산 (Explained Variance):")
print(pca.explained_variance_)

# 주성분의 설명된 분산 비율 출력
print("\n주성분의 설명된 분산 비율 (Explained Variance Ratio):")
print(pca.explained_variance_ratio_)

plt.plot(accmulate.index,accmulate.iloc[:,0])
plt.xlabel("Number of PCA")
plt.ylabel("Cumulative Explained Variance")
plt.show()