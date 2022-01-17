import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data_df = pd.read_csv('data.csv')
test_df = pd.read_csv('test.csv')

kmeans = KMeans(n_clusters=5)
kmeans = kmeans.fit(data_df)

kmeans_label = kmeans.labels_


test_amount = test_df['0'].size
ans = np.empty(test_amount, dtype=int)

for i in range(test_amount):
    print('test case:%d' %(i))
    if kmeans_label[test_df.at[i,'0']]==kmeans_label[test_df.at[i,'1']]:
        ans[i] =1
    else :
        ans[i] =0
    print('answer:%d' %(ans[i]))


ans=pd.DataFrame(ans)
ans.index.name = 'id'
ans.index = ans.index.astype(float)
ans.to_csv('submit.csv',header=['ans'])
