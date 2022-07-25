
# Source 
## GA   解說 https://dotblogs.com.tw/dragon229/2013/01/03/86692
## 供單排程優化 https://medium.com/qiubingcheng/%E4%BB%A5python%E5%AF%A6%E4%BD%9C%E5%9F%BA%E5%9B%A0%E6%BC%94%E7%AE%97%E6%B3%95-genetic-algorithm-ga-%E4%B8%A6%E8%A7%A3%E6%B1%BA%E5%B7%A5%E4%BD%9C%E6%8C%87%E6%B4%BE%E5%95%8F%E9%A1%8C-job-assignment-problem-jap-b0d7c4ad6d0f
## DEAP 文件 https://www.1ju.org/ai-with-python/ai-with-python-genetic-algorithms
## DEAP TSP https://ithelp.ithome.com.tw/articles/10262059
## DEAP 使用與簡介 https://zhuanlan.zhihu.com/p/436438875

## DEAP OneMax example
- https://deap.readthedocs.io/en/master/examples/index.html#genetic-algorithm-ga
- https://zhuanlan.zhihu.com/p/436424904
- https://blog.csdn.net/weixin_42028364/article/details/81539117
# Deap 安裝
## error 
- 遇到pip instapp deap (error: metadata-generation-failed) 降版本 => install setuptools==57.0.0


# 何謂基因演算法

依具生物環境適者生存的特性，模擬出來的演算法，利用選擇、組合、突變的方式找出更優良的基因組合。

我們可以將基因理解程一種解法的組合，透過自訂的適應函數與適應度，找出問題的次佳解，基本上此類演算法為解決<font style='color:deeppink'>組合最佳化問題的工具</font>。


# 為何需要基因演算法
當組合最佳化問題很小的時候，可以透過暴力解找出最佳組合沒問題，但當組合過大需要全部可行的解在有限的時間計算完畢有困難時，基因演算就可做為解決這種問題的工具，此類問題通常為NP問題。值得注意的是基因演算法找到的解不一定為最佳解

# 流程
1. <font style='color:deeppink'>亂數N個基因</font>做為群體
2. 利用<font style='color:deeppink'>適應函數</font>計算適應值(這時應該可排序出，該問題概念上的好壞)
3. 依據每個基因的適應程度，做出篩選
4. 對活下來的基因進行組合、突變，產出更優良的下一代


## * 基因規劃種類
    - 非線性規劃 (例如:圖像還原，投資組合)
    - 整數規劃 (例如:TSP、JAP)
    ---
    - 二進制 : 函示求解、圖像還原(各類使用與否)
    - 整數型 :tsp
# 選擇複製

某種情境下的組合，可能都是由特定的組合構成，
好吃 + 好吃 = 不太可能不好吃!? ，日料好吃又貴的食材往往都是幾種組合單配出來的，但還是有可能臭臭的東西有人以歡所以也不能放棄

## 選擇種類

- 隨機選擇 
- 分裂選擇
  - 方向 (選擇特定方向)
  - 分裂 (只選頭尾)
  - 平衡 (平衡只選中段)
  - 名人選擇 (保留最好的)
- 機率選擇
    - 輪盤式 (依比例分配)
    - 競爭式 (隨機兩個出來比)
    - 等級輪盤 

# 交配

互換基因，但不是每個都會發生，可透過機率去設定

## 組合交配

- 單點交配 (固定的一部分進行互換)
- 雙點交配 (一個區間的互換)
- 多點交配 (多個區間的互換)
- Mask (制定條件進行交配)

## ps.

TSP問題並不是用於單點變異，會導致路徑重複，可使用交換變異。

有些基因初始群會先進行crossover、Mutate，再進行fitness與selection，用意在於初始時產生更多解，擴大可能性