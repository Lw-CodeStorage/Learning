

## 基因長度

最後的結果長度並不會改變，而是在一個固定長度下，得到最優排列組合

# 步驟紀錄
1. 聚落 ; 收集N群祖先聚落
2. 組合 ; 執行基因互換(這部分因業務需求會有所不同)
3. 變異 ; 隨機選取染色體(擷取一段打亂染色體，像是自體變異，避免近親缺點保留)
4. Fitness；與聚落中適合度比較低的，替換聚落中的資料

# Source 
## GA   解說 https://dotblogs.com.tw/dragon229/2013/01/03/86692
## DEAP 文件 https://www.1ju.org/ai-with-python/ai-with-python-genetic-algorithms
## DEAP TSP https://ithelp.ithome.com.tw/articles/10262059
## DEAP 使用與簡介 https://zhuanlan.zhihu.com/p/436438875

## DEAP OneMax example
- https://zhuanlan.zhihu.com/p/436424904
- https://blog.csdn.net/weixin_42028364/article/details/81539117
# Deap 安裝
## error 
- 遇到pip instapp deap (error: metadata-generation-failed) 降版本 => install setuptools==57.0.0