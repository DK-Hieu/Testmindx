
import pandas as pd
# import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import math
from sklearn.linear_model import LinearRegression #mô hình hồi quy tuyến tính
import statistics as sta

print('Completed import lib')


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
print('Completed import')

 
# # Xử lý data


# customer_data = pd.read_excel('D:/Mindx-Career-Kick-start/DA_TEST_1(test2)/customers_data.xlsx',sheet_name='customer info')
# items_data = pd.read_excel('D:/Mindx-Career-Kick-start/DA_TEST_1(test2)/customers_data.xlsx',sheet_name='Items')
# customer_transactions = pd.read_excel('D:/Mindx-Career-Kick-start/DA_TEST_1(test2)/customers_data.xlsx',sheet_name='Customer transactions')

customer_data = pd.read_excel('customers_data.xlsx',sheet_name='customer info')
items_data = pd.read_excel('customers_data.xlsx',sheet_name='Items')
customer_transactions = pd.read_excel('customers_data.xlsx',sheet_name='Customer transactions')


os.chdir('../DA_TEST_1(test2)/Traffic')

extension = 'xlsx'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
traffic_data = pd.concat([pd.read_excel(f) for f in all_filenames ])
traffic_data


traffic_data.info()


customer_data


items_data


customer_transactions


sell = pd.merge(customer_transactions, items_data[['ItemID','Brand','SellPrice','CostPrice']], on='ItemID',how='left')
sell['month'] = pd.DatetimeIndex(sell['TransactionDate']).month
sell['year'] = pd.DatetimeIndex(sell['TransactionDate']).year
sell


 
# # 1.	Website traffic có ảnh hưởng đến doanh thu không ?

 
# ### Lượng view và user sử dụng trong năm 2020


traffic_data['date'] = pd.to_datetime(traffic_data['Posted On (DD/MM/YYYY)']).dt.date
traffic_data_2 = traffic_data.groupby(['date'])[['users','uniquePageviews','pageviews','Brand','date']].sum()
traffic_data_2.reset_index(inplace=True)
traffic_data_2


fig, ax = plt.subplots(figsize=(20,10))
fig.patch.set_facecolor('w')
xs = traffic_data_2['date']
y_users = traffic_data_2['users']
y_uniview = traffic_data_2['uniquePageviews']
y_pageview = traffic_data_2['pageviews']

plt.plot(xs,y_users, label='users')
plt.plot(xs,y_uniview, label='uniquePageviews')
plt.plot(xs,y_pageview, label='pageviews')


plt.title('Lượng view và user sử dụng trong năm 2020')
plt.legend()
plt.show()




 
# ### Doanh thu bán hàng online và bán hàng tại chỗ


revenus_year = sell.groupby('Channel')[['SellPrice']].sum()
revenus_year.loc['Column_Total']= revenus_year.sum(numeric_only=True, axis=0)
revenus_year.reset_index(inplace=True)
revenus_year['change%'] = (revenus_year['SellPrice']/int(revenus_year['SellPrice'].iloc[2:3]))*100
revenus_year


fig = plt.figure(figsize=(10,10))
fig.patch.set_facecolor('w')
sns.barplot(x='Channel', y='change%', data=revenus_year.loc[revenus_year['Channel'] != 'Column_Total'],palette='Blues_d')

 
# Tại năm 2020 doanh thu từ bán hàng online chiếm 49.9% tổng doanh thu của nhãn hàng, cho thấy doanh thu từ bán hàng online có sự ảnh hưởng rất lớn đến doanh nghiệp


sell_month = sell.copy()
revenus_month = sell_month.groupby(['Channel','month'])[['SellPrice']].sum()
revenus_month.reset_index(inplace=True)
revenus_month


fig = plt.figure(figsize=(25,10))
fig.patch.set_facecolor('w')
sns.barplot(x='month', y='SellPrice', data=revenus_month,hue='Channel',palette='Blues_d')

 
# Doanh thu trung bình từ bán hàng truyền thống cao hơn doanh thu từ bán hàng online vào thời gian nghỉ lễ cuối và đầu năm. Giai đoạn giữa năm doanh thu trung bình không chênh lệch quá nhiều.

 
# # 2.	Sản phẩm nào mang lại pageviews và doanh thu


items_view = traffic_data.groupby(['Brand'])[['pageviews','uniquePageviews']].sum()
items_view.sort_values(by='pageviews', ascending=False, inplace= True)
items_view['view_rank'] = items_view['pageviews'].rank(ascending=False)
items_view.reset_index(inplace=True)
items_view


items_res = sell.groupby(['Brand'])[['SellPrice']].sum()
items_res.sort_values(by='SellPrice', ascending=False, inplace= True)
items_res['sell_rank'] = items_res['SellPrice'].rank(ascending=False)
items_res = pd.merge(items_res,items_view,on='Brand',how='left')
items_res.reset_index(inplace=True,drop=True)
items_res


items_res_view = items_res.sort_values(by='pageviews', ascending=False)
items_res_view.reset_index(inplace=True,drop=True)
items_res_view = items_res_view[['Brand','pageviews','SellPrice','view_rank','sell_rank']].head(10)
items_res_view


fig, ax = plt.subplots(facecolor='w')
items_res_view.plot(x='Brand', y=['SellPrice', 'pageviews'], kind='bar',ax=ax ,figsize=(15,10),color=['skyblue','steelblue'], label=['Sell resvenue','Page view'])

# bỏ đường kẻ ở trên và bên phải chart
spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False)

plt.suptitle('Top 10 sản phẩm có page view cao nhất', fontweight='bold', size=14, horizontalalignment='left', x=0.1, y =0.95)


items_res_sell = items_res[['Brand','SellPrice','pageviews','sell_rank','view_rank']].head(10)
items_res_sell


fig, ax = plt.subplots(facecolor='w')
items_res_sell.plot(x='Brand', y=['SellPrice', 'pageviews'], kind='bar',ax=ax ,figsize=(15,10),color=['skyblue','steelblue'], label=['Sell resvenue','Page view'])

# bỏ đường kẻ ở trên và bên phải chart
spines = ['top', 'right']
for s in spines:
    ax.spines[s].set_visible(False)

plt.suptitle('Top 10 sản phẩm có doanh thu bán cao nhất', fontweight='bold', size=14, horizontalalignment='left', x=0.1, y =0.95)


items_res.loc[items_res['pageviews'].isnull() == True]


items_res.isnull().sum()

 
# # 3. Có những phân khúc khách hàng nào? 


data = customer_transactions.merge(customer_data,how='inner',left_on='CustomerID', right_on='ID').merge(items_data,how='inner',left_on='ItemID', right_on='ItemID')
data.drop('ID', inplace=True, axis=1)
data


cus_data = data.groupby(['CustomerID','FirstName', 'LastName','Country','Birthday','DateJoined','Newsletter','Gender']).\
                            agg(total_expenditures = ('SellPrice','sum')).\
                            reset_index()
cus_data


from datetime import date

today = date.today()

# tính tuổi
cus_data['Age'] = range(len(cus_data['Birthday']))
for i in range(len(cus_data['Birthday'])):
    cus_data['Age'][i] = today.year - cus_data['Birthday'][i].year

# tính số năm trở thành thành viên
cus_data['Loyalty'] = range(len(cus_data['DateJoined']))
for i in range(len(cus_data['DateJoined'])):
    cus_data['Loyalty'][i] = today.year - cus_data['DateJoined'][i].year

cus_data


list(set(cus_data['Country']))
list(set(cus_data['Newsletter']))
list(set(cus_data['Gender']))
list(set(cus_data['Loyalty']))
print(list(set(cus_data['Age'])))


list(set(cus_data['Country']))
cus_data.groupby(['Country','Gender'])['Country'].count()


cus_data['Age'].describe()
cus_data.groupby(['Country','Gender'])['Age'].mean()
sns.boxplot(y= cus_data['Age'])


cus_data['total_expenditures'].describe()
cus_data.groupby(['Country','Gender'])['total_expenditures'].sum()
sns.boxplot(y= cus_data['total_expenditures'])

 
# - Có thể thấy những biến Age, Country và Gender có ảnh hưởng đến tổng chỉ tiêu của khách hàng. 
#     + Khách hàng đến từ Đức nhiều hơn các nước khác, và khách nam nhiều hơn. Khách hàng đến từ Mỹ ít nhất và khách hàng nam nữ như nhau.
#     + Độ tuổi trung bình giữa các nước, giữa nam và nữ chênh lệch không lớn (5 năm), độ tuổi trung bình trong tầm 3x
#     + Tổng chi tiêu giữa các nước và giữa nam nữ có sự chênh lệch.
# - Các biến Country, Age, Gender, total_expenditures sẽ được sử dụng để xây dựng model


cus_data['Country_data'] = cus_data['Country'].replace(['UK - United Kingdom','FR - France','IT - Italy','GER - Germany'],[1,2,3,4])
cus_data['Gender_data'] = cus_data['Gender'].replace(['M','F'],[1,2])
cus_data

 
# #### Phân loại khách hàng theo độ tuổi
#     - Nhóm 1: 12-23: Nhóm trẻ
#     - Nhóm 2: 23-33: Nhóm thanh niên
#     - Nhóm 3: 33-45: Nhóm trung niên 
#     - Nhóm 4: 45-57: Nhóm trung niên 2
# 
# #### Phân loại khách hàng theo chi tiêu
#     - Nhóm 1: 745-14790: tiêu dùng ít
#     - Nhóm 2: 14790-25849: tiêu dùng trung bình
#     - Nhóm 3: 25849-36926: tiêu dùng thường
#     - Nhóm 4: 36926-55717: tiêu dùng nhiều


conditions = [
        (cus_data['Age'] > 12) & (cus_data['Age'] <= 23),
        (cus_data['Age'] > 23) & (cus_data['Age'] <= 33),
        (cus_data['Age'] > 33) & (cus_data['Age'] <= 45),
        (cus_data['Age'] > 45) & (cus_data['Age'] <= 57)
        ]

values = [1, 2, 3, 4]
cus_data['Age_data'] = np.select(conditions, values)

conditions = [
    (cus_data['total_expenditures'] > 745) & (cus_data['total_expenditures'] <= 14790),
    (cus_data['total_expenditures'] > 14790) & (cus_data['total_expenditures'] <= 25849),
    (cus_data['total_expenditures'] > 25849) & (cus_data['total_expenditures'] <= 36926),
    (cus_data['total_expenditures'] > 36926) & (cus_data['total_expenditures'] <= 55717)
    ]
    
values = [1, 2, 3, 4]
cus_data['total_expen_data'] = np.select(conditions, values)

cus_data


df = cus_data[['Country_data','Gender_data','Age_data','total_expen_data']]
df

 
# ## Build Model Clustering


from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

 
# ### elbow test


kmeans = KMeans(init='random',n_clusters=3,n_init=10,max_iter=300,random_state=42)
kmeans.fit(df)


kmeans_kwargs = {'init': 'random','n_init': 10,'max_iter': 300,'random_state': 42}
  
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)


fig, ax = plt.subplots(figsize=(20,10))
fig.patch.set_facecolor('w')

plt.style.use('fivethirtyeight')
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()


kl = KneeLocator(range(1, 11), sse, curve='convex', direction='decreasing')
kl.elbow

 
# => Có 4 clusters, như vậy có 4 phân khúc khách hàng

 
# ### Chạy model với k=4


nclusters = 4
kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(df)


df['Cluster Labels'] = kmeans.labels_
df


cus_data['Cluster Labels'] = kmeans.labels_


cus_data['Cluster Labels'].value_counts()


# Phân khúc 1
cluster_1 = cus_data[cus_data["Cluster Labels"] == 0]
cluster_1.head(10)


# Phân khúc 2
cluster_2 = cus_data[cus_data["Cluster Labels"] == 1]
cluster_2.head(10)


# Phân khúc 3
cluster_3 = cus_data[cus_data["Cluster Labels"] == 2]
cluster_3.head(10)


# Phân khúc 4
cluster_4 = cus_data[cus_data["Cluster Labels"] == 3]
cluster_4.head(10)

 
# cmap=plt.hot()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

x = cus_data['total_expen_data']
y = cus_data['Age_data']
z = cus_data['Gender_data']
c = cus_data['Country_data']

img = ax.scatter(x, y, z, c=c)
fig.colorbar(img)
plt.show()


 
# #### Phân khúc khách hàng:
#     - Phân khúc 1:
#         + Khách hàng chủ yếu đến từ Đức
#         + Chủ yếu là nam giới
#         + Khách hàng đa phần thuộc nhóm trung niên thuộc 2 nhóm tuổi (33-45),(45-57)
#         + Là nhóm khách hàng tiêu dùng nhiều nhất.
#     
#     - Phân khúc 2: 
#         + Khách hàng đến từ Pháp và Mỹ
#         + Chủ yếu là Nữ giới
#         + Khách hàng thuộc nhóm khách hàng trẻ và trưởng thảnh
#         + Mức độ tiêu dùng của nhóm khách hàng này rất đa dạng, khách hàng thuộc nhóm trẻ và trưởng thành tiêu dùng nhiều nhất.
#     
#     - Phân khúc 3: 
#         + Khách hàng đến từ Đức và Ý, chủ yếu đến từ Đức
#         + Chủ yếu là Nam giới
#         + Khách hàng thuộc nhóm khách hàng trẻ
#         + Mức độ tiêu dùng của nhóm khách hàng ở mức bình thường.
# 
#     - Phân khúc 4: 
#         + Khách hàng đến từ 4 nước, chủ yếu đến từ Đức và Pháp
#         + Chủ yếu là Nữ giới
#         + Khách hàng thuộc nhóm khách hàng trung niên
#         + Mức độ tiêu dùng của nhóm khách hàng ở mức thấp.        
#     





