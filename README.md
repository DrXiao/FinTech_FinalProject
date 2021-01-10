# FinTech Final Project

* https://github.com/DrXiao/FinTech_FinalProject.git

## KNN 使用原則

* Lump-Sum 手段為準則

* 新的一筆資料進來，ReturnMean_year_Label = -1
    * 如果手中沒有此公司的股票，則不買

    * 持有此公司股票，立馬脫手


* 若是 ReturnMean_year_Label = 1，則判斷...
    * 已經投資四家公司的股，不買

    * 不存在於200大公司中，也不買

    * 存在於200大公司中
        * 手中若持有此公司的股份，且價值高於當初買時，則賣
        * 手中沒有公司股份，價格OK時，則買