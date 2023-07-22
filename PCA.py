import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')
# train = pd.read_csv('dataset/train.csv')
# test = pd.read_csv('dataset/test.csv')

# # PCA 객체 생성 (주성분의 개수를 지정하지 않으면 자동으로 모든 주성분을 유지합니다)
# pca = PCA()
#
# # PCA 모델 훈련
# pca.fit(train)
#
# # 변환된 데이터 얻기 (주성분으로 데이터를 변환)
# transformed_data = pca.transform(train)
#
# # 주성분의 정보 출력
# print("주성분 (Principal Components):")
# print(pca.components_)
#
# # 주성분의 설명된 분산 출력
# print("\n주성분의 설명된 분산 (Explained Variance):")
# print(pca.explained_variance_)
#
# # 주성분의 설명된 분산 비율 출력
# print("\n주성분의 설명된 분산 비율 (Explained Variance Ratio):")
# print(pca.explained_variance_ratio_)
#
# plt.plot(accmulate.index,accmulate.iloc[:,0])
# plt.xlabel("Number of PCA")
# plt.ylabel("Cumulative Explained Variance")
# plt.show()

# In[9]:


train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')




train_X = train.iloc[:,4:]
train_Y = train.iloc[:,0:4]
test_X = test.iloc[:,1:]


pca = PCA()
pca.fit(train_X)


train_score = pca.transform(train_X)
test_score = pca.transform(test_X)

# print(train_X.shape, test_X.shape)
# print(train_score.shape, test_score.shape)

pd.DataFrame(pca.explained_variance_ratio_).plot(kind='bar', figsize=(20,5))

# plt.plot(accmulate.index,accmulate.iloc[:,0])
plt.xlabel("Number of PCA")
plt.ylabel("Cumulative Explained Variance")
plt.show()

exit()
for i in range(227):
    print(i, " : ", pca.explained_variance_ratio_[:i].sum())
'''
0  :  0.0
1  :  0.08372016611255
2  :  0.16727988419077933
3  :  0.24443962821684098
4  :  0.3212978794277497
5  :  0.3885684755863957
6  :  0.454453983122841
7  :  0.5180773374808366
8  :  0.5775905059531596
9  :  0.6352335802823009
10  :  0.6912210012083758
11  :  0.7396754407899901
12  :  0.7857093426069918
13  :  0.8282295686991834
14  :  0.8646967468244922
15  :  0.8934384090242729
16  :  0.9199758429255722
17  :  0.9390722165452405
18  :  0.9527154646678478
19  :  0.9620043307204702
20  :  0.968679779977607
21  :  0.973934166400673
22  :  0.9781727038758014
23  :  0.9817344874510509
24  :  0.984599282680833
25  :  0.9870326909626673
26  :  0.9890798964928219
27  :  0.9907223936828474
'''