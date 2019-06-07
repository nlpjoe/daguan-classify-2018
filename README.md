# 达观杯2018
[![Backers on Open Collective](https://opencollective.com/daguan-2018/backers/badge.svg)](#backers)
 [![Sponsors on Open Collective](https://opencollective.com/daguan-2018/sponsors/badge.svg)](#sponsors) 

参数没调好，仓促比赛，单模型线上没测过，线下0.784，最终得分0.791，排名18/3462，排名不高就不多写了，等着前排分享。思路如同代码所写，很简单。

数据请在[达观数据](http://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E8%B5%9B%E4%BD%93%E4%B8%8E%E6%95%B0%E6%8D%AE.html)处下载，放在data目录下。

### 一、环境

|环境/库|版本|
|:---------:|----------|
|Ubuntu|14.04.5 LTS|
|python|3.6|
|jupyter notebook|4.2.3|
|tensorflow-gpu|1.10.1|
|numpy|1.14.1|
|pandas|0.23.0|
|matplotlib|2.2.2|
|gensim|3.5.0|
|tqdm|4.24.0|


### 二、数据预处理

都写在`jupyter`里了。运行`src/preprocess/EDA.ipynb`生成各种文件。

### 三、baseline模型训练

在`src/preprocess/`中运行：

```
python baseline-x-cv.py
```

### 四、深度模型训练

然后直接train模型，单GPU运行，模型自选：

```
python train_predict.py --gpu 4 --option 5 --model convlstm --feature char
```

多GPU训练示例：

```
python train_predict.py --gpu 4,5,6,7 --option 5 --model convlstm --feature char
```

### 五、模型融合输出

```
python stacking.py --gpu 1 --tfidf True --option 5
```

这里是stacking和伪标签一起做了，请修改代码自选是否用伪标签。




## Contributors

This project exists thanks to all the people who contribute. [[Contribute](CONTRIBUTING.md)].
<a href="https://github.com/nlpjoe/daguan-classify-2018/graphs/contributors"><img src="https://opencollective.com/daguan-2018/contributors.svg?width=890&button=false" /></a>


## Backers

Thank you to all our backers! 🙏 [[Become a backer](https://opencollective.com/daguan-2018#backer)]

<a href="https://opencollective.com/daguan-2018#backers" target="_blank"><img src="https://opencollective.com/daguan-2018/backers.svg?width=890"></a>


## Sponsors

Support this project by becoming a sponsor. Your logo will show up here with a link to your website. [[Become a sponsor](https://opencollective.com/daguan-2018#sponsor)]

<a href="https://opencollective.com/daguan-2018/sponsor/0/website" target="_blank"><img src="https://opencollective.com/daguan-2018/sponsor/0/avatar.svg"></a>
<a href="https://opencollective.com/daguan-2018/sponsor/1/website" target="_blank"><img src="https://opencollective.com/daguan-2018/sponsor/1/avatar.svg"></a>
<a href="https://opencollective.com/daguan-2018/sponsor/2/website" target="_blank"><img src="https://opencollective.com/daguan-2018/sponsor/2/avatar.svg"></a>
<a href="https://opencollective.com/daguan-2018/sponsor/3/website" target="_blank"><img src="https://opencollective.com/daguan-2018/sponsor/3/avatar.svg"></a>
<a href="https://opencollective.com/daguan-2018/sponsor/4/website" target="_blank"><img src="https://opencollective.com/daguan-2018/sponsor/4/avatar.svg"></a>
<a href="https://opencollective.com/daguan-2018/sponsor/5/website" target="_blank"><img src="https://opencollective.com/daguan-2018/sponsor/5/avatar.svg"></a>
<a href="https://opencollective.com/daguan-2018/sponsor/6/website" target="_blank"><img src="https://opencollective.com/daguan-2018/sponsor/6/avatar.svg"></a>
<a href="https://opencollective.com/daguan-2018/sponsor/7/website" target="_blank"><img src="https://opencollective.com/daguan-2018/sponsor/7/avatar.svg"></a>
<a href="https://opencollective.com/daguan-2018/sponsor/8/website" target="_blank"><img src="https://opencollective.com/daguan-2018/sponsor/8/avatar.svg"></a>
<a href="https://opencollective.com/daguan-2018/sponsor/9/website" target="_blank"><img src="https://opencollective.com/daguan-2018/sponsor/9/avatar.svg"></a>


