# 手動加入環境變數  
    - 變數路徑指向Scripts 可直接於 cmd 中使用Scipts安裝過的指令
    - 變數路徑指向pyton.exe 可直接於 cmd 執行python
    - 亦或是直接使用 python -m pip 安裝套件
# 使用virtualenv 創建虛擬環境
    1. 於root python 安裝 virtialenv
    2. 建立一個存放虛擬環境資料夾(任意位置)，之後進入該資料夾內
    3. 建立虛擬環境 cmd python -m venv '虛擬環境變數名稱'
    4. 進入建立完成的虛擬環境 => ./'虛擬環境變數名稱'/Scripts 輸入activate 
# 將虛擬環境整合進入 jupyter note 
    *jupyter kernelspec list 查詢 jupyter kernel list 
    1. 進入新建的虛擬環境Scripts CMD pip install ipykernel
    2. 加入jupyter kernell 中 python -m ipykernel install --user --name=v03
