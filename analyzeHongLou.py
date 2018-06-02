import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import jieba
from wordcloud import WordCloud,ImageColorGenerator
from scipy.ndimage import imread

## 设置字体
font = FontProperties(fname = "C:\Windows\Fonts\msyhbd.ttc",size = 14)
## 设置pandas显示方式
pd.set_option("display.max_rows",8)



#读取停用词
stopword = pd.read_csv("stopword.txt", header=None,names=["Stopwords"], engine="python",sep='\n',error_bad_lines=False)
#读取词典 人物，地点
mydict = pd.read_csv("vocabularys.txt", header=None, names=["Dictionary"], engine="python",sep='\n', error_bad_lines=False,encoding='utf-8')
print(stopword)
print("------------------------------")
print(mydict)
#红楼梦的内容
RedDream = pd.read_csv("honglou.txt", header=None, names=["Reddream"], engine="python", error_bad_lines=False)
print(RedDream)
print(RedDream.Reddream[100])

#去除空行
print(np.sum(pd.isnull(RedDream)))

#使用正则表达式对数据进行预处理

## 找出每一章节的头部索引和尾部索引
## 每一章节的名字
indexhui = RedDream.Reddream.str.match("^第+.+回")
chapnames = RedDream.Reddream[indexhui].reset_index(drop=True)
print(chapnames)
print("---------------------------")

## 处理章节名，使用空格分割字符串
chapnamesplit = chapnames.str.split(" ").reset_index(drop=True)
print(chapnamesplit)

## 建立保存数据的数据表
Red_df = pd.DataFrame(list(chapnamesplit), columns=["Chapter","Leftname","Rightname"])
## 添加新的变量
Red_df["Chapter2"] = np.arange(1,121)
Red_df["ChapName"] = Red_df.Leftname+","+Red_df.Rightname

## 每章的开始行（段）索引
Red_df["StartCid"] = indexhui[indexhui == True].index

## 每章的结束行数
Red_df["endCid"] = Red_df["StartCid"][1:len(Red_df["StartCid"])].reset_index(drop = True) - 1
Red_df["endCid"][[len(Red_df["endCid"])-1]] = RedDream.index[-1]

## 每章的段落长度
Red_df["Lengthchaps"] = Red_df.endCid - Red_df.StartCid
Red_df["Artical"] = "Artical"

## 每章节的内容
for ii in Red_df.index:
    ## 将内容使用句号连接
    chapid = np.arange(Red_df.StartCid[ii]+1,int(Red_df.endCid[ii]))
    ## 每章节的内容，
    Red_df["Artical"][ii] = "".join(list(RedDream.Reddream[chapid])).replace("\u3000","")
##计算某章节的内容
aa = np.arange(Red_df.StartCid[0]+1,int(Red_df.endCid[0]))
len("".join(list(RedDream.Reddream[aa])).replace("\u3000",""))
Red_df["lenzi"] = Red_df.Artical.apply(len)
print(Red_df)

## 字长和段落长的散点图
plt.figure(figsize=(8,6))
plt.scatter(Red_df.Lengthchaps,Red_df.lenzi)
for ii in Red_df.index:
    plt.text(Red_df.Lengthchaps[ii]+1,Red_df.lenzi[ii],Red_df.Chapter2[ii])
plt.xlabel("章节段数",FontProperties = font)
plt.ylabel("章节字数",FontProperties = font)
plt.title("《红楼梦》120回",FontProperties = font)
plt.show()
##章节段数和章节字数的折线图
plt.figure(figsize=(12,10))
plt.subplot(2,1,1)
plt.plot(Red_df.Chapter2,Red_df.Lengthchaps,"ro-",label = "段落")
plt.ylabel("章节段数",FontProperties = font)
plt.title("《红楼梦》120回",FontProperties = font)
## 添加平均值
plt.hlines(np.mean(Red_df.Lengthchaps),-5,125,"b")
plt.xlim((-5,125))

plt.subplot(2,1,2)
plt.plot(Red_df.Chapter2,Red_df.lenzi,"ro-",label = "段落")
plt.xlabel("章节",FontProperties = font)
plt.ylabel("章节字数",FontProperties = font)
## 添加平均值
plt.hlines(np.mean(Red_df.lenzi),-5,125,"b")
plt.xlim((-5,125))
plt.show()

## 将红楼梦每一章节的内容提取出来
Redcontent = Red_df.Artical
print(Redcontent)
print("------------------------------")
print(RedDream.Reddream[20])
## 分词后返回的是一个generator 发生器
list1 = jieba.cut(RedDream.Reddream[20], cut_all=True)
## 查看分词内容
print("/".join(list1))

## 添加自定义词典定义一些地名和人名，使分词更准确
jieba.load_userdict("vocabularys.txt")
list2 = jieba.cut(RedDream.Reddream[20], cut_all=True)
print("/".join(list2))

#提取分词后的关键词

## 提取长度大于1的词，精准查询
list3 = jieba.cut(Redcontent[2], cut_all=True)
word = []
for li in list3:
    if len(li) > 1:
        word.append(li)

## 构建数据表
word_df = pd.DataFrame({"Word":word})
print(word_df)

## 去停用词统计词频
word_df = word_df[~word_df.Word.isin(stopword)]
print(len(word_df))
word_stat = word_df.groupby(by=["Word"])["Word"].agg({"number":np.size})
word_stat = word_stat.reset_index().sort_values(by="number",ascending=False)
print(word_stat)

##对全文进行分词
## 数据表的行数
row,col = Red_df.shape
## 预定义列表
Red_df["cutword"] = "cutword"

for ii in np.arange(row):
    jieba.load_userdict("vocabularys.txt")
    ## 分词
    cutwords = list(jieba.cut(Red_df.Artical[ii], cut_all=True))
    ## 去除长度为1的词
    cutwords = pd.Series(cutwords)[pd.Series(cutwords).apply(len)>1]
    ## 去停用此
    cutwords = cutwords[~cutwords.isin(stopword)]
    Red_df.cutword[ii] = cutwords.values
## 查看最后一段的分词结果
print(cutwords)
print(cutwords.values)
##查看全文的分词结果
print(Red_df.cutword)


##对全文进行词频统计 绘制词云
print(np.concatenate(Red_df.cutword))
words = np.concatenate(Red_df.cutword)
print(words.shape)
## 统计词频
word_df = pd.DataFrame({"Word":words})
word_stat = word_df.groupby(by=["Word"])["Word"].agg({"number":np.size})
word_stat = word_stat.reset_index().sort_values(by="number",ascending=False)
word_stat["wordlen"] = word_stat.Word.apply(len)
print(word_stat)

## 出现次数大于250次的词语数目
print(len(word_stat.loc[word_stat.number > 250]))
## 出现次数大于4次的词语数目
print(len(word_stat.loc[word_stat.number > 4]))

##词云
## 连接全文的词
print("/".join(np.concatenate(Red_df.cutword)))
## width=1800, height=800 设置图片的清晰程度
wlred = WordCloud(font_path="C:\Windows\Fonts\msyhbd.ttc",  ## 显示中文，指定字体
                 width=1800, height=800).generate("/".join(np.concatenate(Red_df.cutword)))
## 显示中文
plt.figure(figsize=(15,10))
plt.imshow(wlred)
plt.show()

##通过generate_from_frequencies()生成词云
##指定每个词语和它对应的频率绘制词云
## 数据准备
worddict = {}
## 构造 词语：频率 字典
for key,value in zip(word_stat.Word,word_stat.number):
    worddict[key] = value
## 查看其中的10个元素
for ii,myword in zip(range(10),worddict.items()):
    print(ii)
    print(myword)
## 生成词云
redcold = WordCloud(font_path="C:\Windows\Fonts\msyhbd.ttc",margin=5, width=1800, height=1800)
redcold.generate_from_frequencies(frequencies=worddict)
plt.figure(figsize=(15,10))
plt.imshow(redcold)
plt.axis("off")
plt.show()

##生成有背景的图片
## 读取背景图片
back_image = imread("./image/shade.jpg")

# 生成词云, 可以用我们计算好词频后使用generate_from_frequencies函数
red_wc = WordCloud(font_path="C:\Windows\Fonts\msyhbd.ttc", ## 设置字体
               margin=5, width=6000, height=4000, ## 字体的清晰度
               background_color="white",  # 背景颜色
               max_words=2000,  # 词云显示的最大词数
               mask=back_image,  # 设置背景图片
               # max_font_size=100, #字体最大值
               random_state=42,
               ).generate_from_frequencies(frequencies=worddict)

# 从背景图片生成颜色值
image_colors = ImageColorGenerator(back_image)

# 绘制词云
plt.figure(figsize=(15,10))
plt.imshow(red_wc.recolor(color_func=image_colors))
plt.axis("off")
plt.show()

##词语出现次数较高的直方图
## 筛选数据
newdata = word_stat.loc[word_stat.number > 500]
print(newdata)

## 绘制直方图
newdata.plot(kind="bar",x="Word",y="number",figsize=(10,7))
plt.xticks(FontProperties = font,size = 9)
plt.xlabel("关键词",FontProperties = font)
plt.ylabel("频数",FontProperties = font)
plt.title("《红楼梦》",FontProperties = font)
plt.show()

## 筛选数据
newdata = word_stat.loc[word_stat.number > 250]
print(newdata)

## 绘制直方图
newdata.plot(kind="bar",x="Word",y="number",figsize=(16,7))
plt.xticks(FontProperties = font,size = 8)
plt.xlabel("关键词",FontProperties = font)
plt.ylabel("频数",FontProperties = font)
plt.title("《红楼梦》",FontProperties = font)
plt.show()

##Red_df.to_json("Red_dream_data.json")

