import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import nltk
from nltk.cluster import KMeansClusterer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.spatial.distance import pdist,squareform
from sklearn.decomposition import PCA

## 设置字体
font = FontProperties(fname="C:\Windows\Fonts\msyhbd.ttc",size = 14)
## 设置pandas显示方式
pd.set_option("display.max_rows",8)

#构建分词结果的TF-IDF矩阵
#tf-idf是一种用于资讯检索与文本挖掘的常用加权技术
#tf-idf是一种统计方法，用以评估对于一个文件集或一个语料库中的一份文件的重要程度。
#字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降
#CountVectorizer()可以将使用空格分开的词整理为语料库

#准备工作，将分词后的结果整理成，CountVectorize() 可用的形式
#将所有分词后的结果使用空格连接为字符串，并组成列表，每一段为列表中的一个元素
Red_df = pd.read_json("Red_dream_data.json")
print(Red_df.cutword)
articals = []
for cutword in Red_df.cutword:
     articals.append(" ".join(cutword))
print(articals)
##构语料库，并计算文档一一词的TF-IDF
## tfidf 以稀疏矩阵的形式存储
##  将tfidf转化为数组的形式,文档－词矩阵
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(articals))
print(tfidf.toarray)
print(tfidf)
dtm = tfidf.toarray()


#使用TF-IDF矩阵对章节进行聚类
## 使用夹角余弦距离进行k均值聚类
#越接近1，夹角越接近0，越相似
kmeans = KMeansClusterer(num_means=3, ## 聚类数目
                         distance=nltk.cluster.util.cosine_distance,## 夹角余弦距离
                        )
kmeans.cluster(dtm)

##  聚类得到的类别
labpre = [kmeans.classify(i) for i in dtm]
kmeanlab = Red_df[["ChapName","Chapter"]]
kmeanlab["cosd_pre"] = labpre
print(kmeanlab)

## 查看每类有多少个分组
count = kmeanlab.groupby("cosd_pre").count()

## 可视化
count = count.reset_index()
count.plot(kind="barh", figsize=(6,5), x="cosd_pre", y="ChapName", legend=False)
for xx,yy,s in zip(count.cosd_pre,count.ChapName,count.ChapName):
  plt.text(y =xx-0.1, x = yy+0.5,s=s)
plt.ylabel("cluster label")
plt.xlabel("number")
plt.show()

mds = MDS(n_components=2,random_state=123)
## 对数据降维
coord = mds.fit_transform(dtm)
print(coord.shape)

## 绘制降维后的结果
plt.figure(figsize=(8,8))
plt.scatter(coord[:,0],coord[:,1],c=kmeanlab.cosd_pre)
for ii in np.arange(120):
    plt.text(coord[ii,0]+0.02,coord[ii,1],s = Red_df.Chapter2[ii])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-means MDS")
plt.show()



# 聚类结果可视化
## 使用PCA对数据进行降维
pca = PCA(n_components=2)
pca.fit(dtm)
print(pca.explained_variance_ratio_)
## 对数据降维
coord = pca.fit_transform(dtm)
print(coord.shape)

## 绘制降维后的结果
plt.figure(figsize=(8,8))
plt.scatter(coord[:,0],coord[:,1],c=kmeanlab.cosd_pre)
for ii in np.arange(120):
    plt.text(coord[ii,0]+0.02,coord[ii,1],s = Red_df.Chapter2[ii])
plt.xlabel("主成分1",FontProperties=font)
plt.ylabel("主成分2",FontProperties=font)
plt.title("K-means PCA")
plt.show()


##层次聚类(Hierarchical Clustering)是聚类算法的一种，
# 通过计算不同类别数据点间的相似度来创建一棵有层次的嵌套聚类树。
# 在聚类树中，不同类别的原始数据点是树的最低层，树的顶层是一个聚类的根节点。
# 创建聚类树有自下而上合并和自上而下分裂两种方法。

## 标签，每个章节的名字
labels = Red_df.Chapter.values
cosin_matrix = squareform(pdist(dtm)) # 计算每章的距离矩阵
ling = ward(cosin_matrix)  ## 根据距离聚类
## 聚类结果可视化
fig, ax = plt.subplots(figsize=(10, 15)) # 设置大小
ax = dendrogram(ling, orientation='right', labels=labels);
plt.yticks(FontProperties=font,size = 8)
plt.title("《红楼梦》各章节层次聚类",FontProperties=font)
plt.tight_layout() # 展示紧凑的绘图布局
plt.show()