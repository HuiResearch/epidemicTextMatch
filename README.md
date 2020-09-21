# “公益AI之星”挑战赛-新冠疫情相似句对判定大赛
天池大赛疫情文本挑战赛线上第三名方案分享
====

欢迎大家使用[tensorflow1.14下调用bert的小工具](https://github.com/huanghuidmml/textToy)

相信所有参赛人员都有种感觉，就是这次比赛的数据质量太好、bert模型太强，
完全找不到下手的地方，其实我也是......我尝试了很多trick，但是有用的就
四个：数据增强、fgm训练、调参以及模型融合。trick之少，可能会让大家有所失望，抱歉。


[比赛链接](https://tianchi.aliyun.com/competition/entrance/231776/information)

**方案介绍**
------
### **任务简介**
本次大赛主要面对疫情抗击，疫情知识问答应用得到普遍推广。是由
达摩院联合医疗服务机构妙健康发布的疫情相似句对判定任务，比赛整理
近万条真实语境下疫情相关的肺炎、支原体肺炎、支气管炎、
上呼吸道感染、肺结核、哮喘、胸膜炎、肺气肿、感冒、咳血等患者提问句对。
而任务就是判定两问句是否相似，属于问句相似性判断任务。
以下是示例句子：

| query1 | query2 | label |
| :------: | :------: | :------: |
| 剧烈运动后咯血,是怎么了?| 剧烈运动后咯血是什么原因？| 1 |
| 剧烈运动后咯血,是怎么了?| 剧烈运动后为什么会咯血？| 1 |
| 剧烈运动后咯血,是怎么了?| 剧烈运动后咯血，应该怎么处理？| 0 |
| 剧烈运动后咯血,是怎么了?| 剧烈运动后咯血，需要就医吗？| 0 |
| 剧烈运动后咯血,是怎么了?| 剧烈运动后咯血，是否很严重？| 0 |

本次任务我采用的方式和大家一样，bert+fc，构造句子对输入，
直接送入bert进行0 1分类。

### **任务难点**
姑且写一个稍微正经点的难点，其实最大的难点就是我上边说的......
* **数据量小**


数据量小一般就会造成模型容易过拟合或欠拟合。
### **解决方案**

#### **数据增强**
在比赛过程中我采用过很多数据增强，例如bert mask预测进行关键词替换（整理后的代码没有
提交这些失败的方案，做法可以借鉴[华为开源的tinyBert数据增强方法。](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/data_augmentation.py)）,
还有eda，ICD医疗词汇替换增强等，均让我线下效果下降了，也就放弃了进行线上测试。
有用的数据增强就是数据传递，例如A与B相似，A与C相似，A与D不相似，因此得到B、C相似，B、C和D不相似，
但同时要注意，数据增强不能过多，需要保证扩充的数据0 1标签分布要和原数据一致，这样才能提升效果，
详情方法见[增强代码部分](code/data_aug.py)。我稍微测了下线上，数据增强后ernie五折结果提升了3-4个千分点。

#### **fgm**
增加对抗训练应该算是一个通用的提升模型方法，可以有效提高模型的鲁棒性和泛化能力，
详情参见[功守道：NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)。
本次代码中只需要将train.py中的adv_type改成fgm就行了。
在fgm中我只对word embedding进行了扰动，尝试过position embedding等的扰动，
但是都下降了。fgm代码在[adv_utils.py](code/utils/adv_utils.py)

#### **调参**
调参真的是一个恶心的过程，但是在bert时代，得调参者得天下......，
参数中最有奇效的是lr和weight decay（0.01），参数设定的话请见[train.py](code/train.py)
和[main.sh](code/main.sh)

#### **模型融合**
模型融合方式我采用的投票，logits取平均的话单模比投票稍低5个万分点，
方式：五折组间投票，然后投票结果再与其他模型进行投票。详细代码见[predict.py](code/predict.py)
最终融合的模型是[roberta_wwm_ext_large](https://github.com/ymcui/Chinese-BERT-wwm)、
[ernie_base](https://github.com/nghuyong/ERNIE-Pytorch)、
[roberta-pair-large](https://github.com/CLUEbenchmark/CLUEPretrainedModels).
请不要喷我融合太多...为了上分，没办法，不融合干不过。

#### **失败trick**

本次比赛，我阵亡的trick比有用的trick多多了......

1. bert mask预测进行关键词替换；
2. 疾病等关键词替换；
3. 自蒸馏；
4. 标签平滑（线下有效，但是线上降了几个万分点）；
5. 取多层bert输出进行各种骚操作；
6. 在bert后增加lstm、gru、attention、cnn、capsule net各种网络，
   并且我在法研杯有效的rcnn加上去也降了。
7. 以及其它我记不清楚的不知名trick。

**代码说明**
-------

#### **基本代码**
开源的代码是我整理来提交复现的，一股脑训练预测运行流程请见[main.sh](code/main.sh)


下边是我线下测试的一个运行流程。


**数据增强**


线下的话只需要增强训练集就好，用验证集测试。
在data_aug中修改main下边的代码。
```
datas = load_data('train.csv路径')
datas = data_aug(datas)  # 进行数据增强操作
write_fold_data(datas, '增强后数据保存文件名')
```
线下测试的话首先把增强后的数据放到train.csv文件夹下，
然后还要将[code/utils/data_utils.py](code/utils/data_utils.py)
中的PairProcessor部分（31行左右）的对应训练数据名称train.csv替换成增强后的训练数据名称。

**训练**

在train.py指定好其余训练参数后。如下运行：
```
CUDA_VISIBLE_DEVICES=0 python train.py \
--do_train \
--do_eval \
--do_eval_during_train \
```
程序会边运行边进行模型筛选，每五十步筛选一次，保存在验证集上acc最好的模型。

**预测**

预测代码是[code/predict.py](code/predict.py)，提交时用法：
1. 将77行vote_model_paths参数替换成你的五折模型文件夹就行了，文件夹下边是五个子文件夹;
    如果有多个模型用英文逗号分隔。
2. 指定好测试集文件名(predict_file)和预测结果文件名(predict_result_file)就行了。
3. 然后 CUDA_VISIBLE_DEVICES=0 python predict.py 完事。

**Reference**
-----
1. [Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch. ](https://github.com/huggingface/transformers)
