# PCIC2021_Track2_Rank1st_Solution
Rank1st Solution of PCIC 2021: Causal Inference and Recommendation 
### PCIC 2021 Track2 Rank1st Solution
##### Huawei Noah's Ark Lab 和 Peking University 联合举办的 PCIC 2021: Causal Inference and Recommendation 已经告一段落，来自深圳福田莲花街道的 Yetta 有幸在初赛、复赛、决赛中均取得了TOP1的成绩，下面给出他的解决方案。
<a href="https://competition.huaweicloud.com/information/1000041488/introduction?track=107" target="_blank">PCIC 2021 竞赛地址</a>
#### 一、背景介绍
<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632489161825052011.PNG" width = 60%/> </div>

在现实世界的电影推荐中，每部电影都与一些标签相关联，如《复仇者联盟》有着科幻，动作，漫威，漫画电影等标签。本次任务着重于预测用户是否会喜欢某个电影标签。这个任务存在着很多bias以及相关数据的缺失，比如用户看的电影有限，很多标签没有接触过；或者用户倾向于选择流行度比较高的电影，缺乏与冷门电影标签交互的数据；或者用户只会关注到推荐系统中推荐位置比较靠前的标签，比如一个电影有八个标签，因为推荐位置的原因用户只注意到了前几个标签，其他标签可能也会喜欢但却没有留下任何相关的记录等等。

<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632489252228001536.PNG" width = 80%/> </div>

本任务提供用户对电影的评级数据（Fig.2 rating），用户对电影标签的标记行为数据（Fig.2 bigtag & choicetag），每部电影与标签的隶属关系（Fig.2 movie），需要预测用户对历史行为数据中没有接触过标签的喜好，此外赛题给出了少量用户对未接触标签喜好的无偏数据作为训练的label，赛题采用AUC作为评估指标。
#### 二、本地验证
<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632489596293088503.PNG" width = 60%/> </div>

为了获得可靠的本地验证策略，减小提交次数，提高模型的泛化能力，我使用了带有shuffle的交叉验证作为本地验证策略，其过程如Fig.3所示。在线下进行数据试验时，使用交叉验证K个验证集的AUC均值作为线下模型评估的结果。

<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632489669495023753.PNG" width = 60%/> </div>

Fig. 4是初赛阶段线上测试集和线下验证集的分数对比，蓝色的是线下交叉验证的结果，红色的是线上测试集的结果，可以看出这个线下验证策略与线上有着较高的一致性。如果线下AUC提升超过0.005，那么线上测试集几乎一定是提升的，但是如果线下提升较小，没有0.005，线上则不一定是上升还是下降。因此本任务所有线下AUC提升小于0.005的idea都可以认为是随机波动，没必要提交到线上去，并且所有提升小于0.005的idea也都不会进入我最终的解决方案，这让我的模型有着很好的泛化能力，在初赛前排选手们复赛普遍掉分的背景下依旧能有着很好的效果。
#### 三、特征工程
赛题任务是预测用户是否会喜欢某个电影标签，因此我们可以从（用户），（电影标签），（用户-电影标签对）这三个角度的信息来处理这个问题。
#### 3.1 用户角度的特征构造
在数据探索阶段，用户角度我设计了很多变量，比如用户标记过多少部不同的电影，标记过多少电影标签，或者用户是不是给每部电影都打高分或者都打低分等等。但是所有用户角度的变量在线下都无法通过本地交叉验证，提升非常小，甚至有些特征集合会让模型效果变得更差。我猜测效果不好的原因可能来自于bias，即用户主动的行为会受到bias的影响，比如用户标记过的电影或标签都是之前推荐系统推荐的或者热门的，或者位置靠前的，而我们需要预测的几乎是用户没有接触过的标签，这导致了相关信息的无效。因此我最终的方案中也没有使用单纯用户相关的变量，但是会使用到一些隐藏在数据之下，更深层次的信息，3.3节会有所介绍。
#### 3.2 标签角度的特征构造
标签相关的特征构造属于较为常规的类别型特征构造，如有多少部电影出现过这个标签（不区分推荐位置），这个标签被多少个用户标记过，该标签出现在所有电影推荐位置1的次数，位置2的次数等等。标签相关的特征大多有着不错的效果。
#### 3.3 用户-标签对的特征构造
<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632496734863077020.PNG" width = 60%/> </div>

Bigtag数据集记录了每个用户对电影标签的标记行为，如Fig.5所示，如果在bigtag中一个用户标记过某个标签，那他在训练集中一定是喜欢这个标签。但待预测的样本几乎完全是用户没有标记过的，如Fig.6所示，训练集中仅有4.02%的标签是用户已标记过的，初赛测试集中这个比例只有1.98%，复赛测试集中这个比例甚至为0%。因此如何拓展出更多的（用户-标签）的信息显得至关重要。

<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632496996540055643.PNG" width = 60%/> </div>

虽然用户主动标记的电影标签很少。但是通过用户对电影标签的标记行为，用户给电影的评分记录，以及电影和标签的隶属关系，我们可以构建出庞大的用户，标签，电影的图网络，从图网络中我们就可以获得用户到某未知标签更多的信息，突破bias带来的信息不全的问题。如Fig.7所示，虽然用户1仅标记过标签1，2，3，但在图网络中我们可以找到用户1到其他未标记过标签的路径。

<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632497113605039132.PNG" width = 70%/> </div>

Fig.8中展示了部分有效路径，比如用户1标记了标签2，而标签2属于电影3，电影3拥有标签7，这样我们就拓展出了用户1与标签7之间的信息；或者用户1喜欢标签1，标签1也被用户2喜欢，用户2喜欢标签10，这样我们就得到了用户1与标签10之间的信息。基于图网络的思路我们就可以很快找到很多特征构造的方案，比如用户1到标签7之间有多少部电影，用户1到标签10之间有多少个共同喜好的中间用户等等。图网络路径很多，因此也有很多构造特征集合的方向，这些特征集合有些可能很强，有些可能很弱，甚至对模型有负面效果。此时我们只需要在选取路径构建完变量集合后，使用第二章的线下验证进行校验即可，在第二章中已知本地交叉验证在AUC提升>0.005时为可靠提升，因此所有线下提升小于0.005的路径都可以认为是无效的噪声路径，仅保留有效的强路径特征集合即可。

<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632497153914092977.PNG" width = 70%/> </div>

#### 3.4 特征筛选
不需要进行额外的特征筛选。
在数据探索的过程中，我会根据每个idea设计出一组特征集合，然后建模进行数据试验，如果这组特征集合无法通过本地验证，就不会参与后续的数据试验。因此在每次试验后我的所有特征都是有用的，不需要再进行额外的特征筛选。

<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632497211867046654.PNG" width = 60%/> </div>

我的整个特征工程的过程如Fig.9所示，除了前文提到的三个方向的特征外，我还尝试了embedding特征和基于数据增强的特征。使用word2vec对电影，用户，标签做embedding，比如对电影标签groupby后，将相关的电影或用户当作词汇拼接起来，再进行word2vec，但词嵌入方向的所有变量均无法带来本地的有效提升。数据增强的角度我也做了多种尝试，比如把模型预测test集中predict score很高或很低的样本拿来单独建模，把单独建模的模型结果用作正常训练集的其中一个变量，或将这些样本与正常样本一起建模等等，但是也都没什么用，最终有用的只有标签相关的统计特征，以及基于图网络路径构建的统计特征。
### 四、模型选择与框架
凭借训练速度快，拟合精度高，泛化能力强等优点，基于GBDT的LightGBM算法框架在2017年被微软开源后便被广泛应用于各类结构化数据的任务中并获得了大量成功应用的案例，比如：
IJCAI 2018 Alimama International Advertising Algorithm Competition
*Rank1: https://github.com/plantsgo/ijcai-2018*
*Rank2:https://github.com/YouChouNoBB/ijcai-18-top2-single-mole-solution*
*Rank3: https://github.com/luoda888/2018-IJCAI-top3*

WSDM 2018 KKBox's Music Recommendation Challenge
*Rank1:https://github.com/lystdo/codes-for-wsdm-cup-music-rec-1st-place-solution*

KDD Cup 2020 Challenges for Modern E-Commerce Platform: Debiasing
*Rank1: https://github.com/aister2020/KDDCUP_2020_Debiasing_1st_Place*

PAKDD 2021 2nd Alibaba Cloud AIOps Competition
*Rank1: https://github.com/ji1ai1/202101-PAKDD2021*
等等。
本赛题中我也采用了LightGBM框架，模型超参数采用祖传参数，并未调参。

<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632497407373091318.PNG" width = 60%/> </div>

最终的模型框架如Fig.10所示，虽然在建模过程中我进行了大量不同idea的数据试验，但真正有用的方向仅有标签相关的变量以及图网络相关的变量，因此我最终的模型框架非常简单，从数据集中构造标签相关的变量以及图网络相关的变量，K折建立一个LightGBM模型，无其他模型融合，直接输出最终预测结果即可。

<div align=center> <img src="https://bbs-img.huaweicloud.com/blogs/img/20210924/1632497446845099933.PNG" width = 60%/> </div>

Fig.11 是最终模型的特征重要性 TOP20。


