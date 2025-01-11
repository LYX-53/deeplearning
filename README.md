### 一、数据集

#### （一）耶拿数据集（jena_climate_2009_2016.csv)

 1.数据来源：时间序列数据集资源库https://gitcode.com/open-source-toolkit/190a9/overview?utm_source=highlight_word_gitcode&word=%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E9%9B%86&isLogin=1

 2.数据总量：40000+

 3.数据示例：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml34060\wps1.jpg) 

 

#### （二）从气象网站爬取广州一周内天气的数据集(weather2.csv)(仅在所设计的混合神经网络模型上进行试水用)

1.数据来源：中国气象网https://weather.cma.cn/web/weather/59287.html

 2.数据总量：56（一周内间隔三小时）

3.数据示例（经过修改规范化）：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml34060\wps2.jpg) 

（weather列0为多云，1为阴，2为小雨，3为中雨）

 

#### （三）柏林数据集(Berlin_updated.csv)

 1.数据来源：open meteo

https://api.open-meteo.com/v1/forecast?latitude=35.6895&longitude=139.6917&hourly=temperature_2m

 2.数据总量：20000+

3.数据示例：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml34060\wps3.jpg) 

 

#### （四）上海数据集(shanghai_updated.csv)

 1.数据来源：open meteo

https://api.open-meteo.com/v1/forecast?latitude=35.6895&longitude=139.6917&hourly=temperature_2m

2.数据总量：10000+

3.数据示例：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml34060\wps4.jpg) 

 

#### （五）广州数据集(guangzhoudata.csv)

1.数据来源：NCDC 提供的开放的 FTP 服务器[ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/](ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/)

2.数据总量：80000+

3.数据示例：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml34060\wps5.jpg)



### 二、模型在相应数据集上训练的代码：

#### 1.本研究所设计的混合神经网络模型(EnhancedHybridModel)

该模型架构如下：

![image-20250112013610106](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250112013610106.png)

(1)对应使用耶拿气象数据集（jena_climate_2009_2016.csv）-14个变量,数据四万多条；

代码为:LCGA_improve.py

(2)对应使用从气象网站爬取广州一周内天气的数据集(weather2.csv)，数据几十条，仅在本模型上进行试水用；

代码为：LCGA_improve_new_data.py

(3)对应使用Berlin_updated.csv（柏林2022.1.1至今）-22个变量,数据两万多条；

代码为：LCGA_improve_new_data_Berlin.py

(4)对应使用上海气象站数据(shanghai_updated.csv)：8个变量，数据一万多条；

代码为：LCGA_improve_new_data_shanghai.py

(5)对应使用广州数据集(guangzhoudata.csv)（8个变量，数据8万多条）

代码为：LCGA_improve_new_data_guangzhou.py

#### 2.使用三种已有的方法：

根据下面三种网络上已有的方法，结合实际数据集进行代码的设计修改

##### 方法一：基于LSTM+注意力机制(self-attention)进行天气变化的时间序列预测

(1)对应使用耶拿气象数据集（jena_climate_2009_2016.csv）-14个变量,数据四万多条；

代码为:Time-weather_change_map.py

(2)对应使用Berlin_updated.csv（柏林2022.1.1至今）-22个变量,数据两万多条；

代码为：Time-weather_Berlin_updated.py

(3)对应使用上海气象站数据(shanghai_updated.csv)：8个变量，数据一万多条；

代码为：Time-weather_shanghai_new_data.py

(4)对应使用广州数据集(guangzhoudata.csv)（8个变量，数据8万多条）

代码为：Time-weather_guangzhoudata.py

##### 方法二：使用GRU（不同的LSTM变体）进行天气变化的时间序列预测

(1)对应使用耶拿气象数据集（jena_climate_2009_2016.csv）-14个变量,数据四万多条；

代码为:GRU_change_map.py

(2)对应使用Berlin_updated.csv（柏林2022.1.1至今）-22个变量,数据两万多条；

代码为：GRU_Berlin_updated.py

(3)对应使用上海气象站数据(shanghai_updated.csv)：8个变量，数据一万多条；

代码为：GRU_shanghai_data.py

(4)对应使用广州数据集(guangzhoudata.csv)（8个变量，数据8万多条）

代码为：GRU_guangzhou_data.py

##### 方法三：CNN-LSTM混合神经网络气温预测

(1)对应使用耶拿气象数据集（jena_climate_2009_2016.csv）-14个变量,数据四万多条；

代码为:CNN-LSTM-jienadatasets-method3-newway.py

(2)对应使用Berlin_updated.csv（柏林2022.1.1至今）-22个变量,数据两万多条；

代码为：CNN-LSTM-BerLindatasets-methods3.py

(3)对应使用上海气象站数据(shanghai_updated.csv)：8个变量，数据一万多条；

代码为：CNN-LSTM-shanghaidatasets-methods3.py

(4)对应使用广州数据集(guangzhoudata.csv)（8个变量，数据8万多条）

代码为：CNN-LSTM-guangzhoudatasets-methods3.py



### 三、开发环境

| 环境配置 |                                                |
| -------- | ---------------------------------------------- |
| Python   | 3.8(ubuntu20.04)                               |
| GPU1     | NVIDIA GeForce RTX 3050 Laptop GPU             |
| CPU      | Intel(R) UHD Graphics                          |
| GPU0     | 11th Gen Intel(R) Core(TM) i5-11260H @ 2.60GH2 |
| PyTorch  | 2.0.0                                          |
| Ubuntu   | 20.04                                          |

