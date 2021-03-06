## 看图说话项目节点报告
- _小组成员：李雨阳，朱怡，张臣生_


_第一周_
### 数据准备
#### 数据下载
````
 flickr8k下载地址:
 http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip
 http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip   
````
````
coco:
http://images.cocodataset.org/zips/train2014.zip
http://images.cocodataset.org/zips/val2014.zip
http://images.cocodataset.org/annotations/annotations_trainval2014.zip
````
#### 数据格式的理解
##### flickr8k
- 原始数据结构如下
  ````
    + Flickr8k_Dataset                    <图片数据，共8000多张图片>
    + Flickr8k_text
      - CrowdFlowerAnnotations.txt        <图片与文字关联度>
      - ExpertAnnotations.txt             <图片与文字关联度>
      - Flickr_8k.devImages.txt           <验证集 共1000张图片，对应5000条训练数据>
      - Flickr_8k.testImages.txt          <测试集 共1000张图片，对应5000条训练数据>
      - Flickr_8k.trainImages.txt         <训练集 共6000张图片，对应30000条训练数据>
      - Flickr8k.lemma.token.txt          <对文字作了 lemmatized 处理>
      - Flickr8k.token.txt                <图片及对应的ground truth>
  ````
- Lemmatized含义理解
  ```
  <raw caption>
  Two motorists are riding along on their vehicle that is oddly designed and colored .
  ```
  ```
  <lemmatized caption>
  Two motorist be ride along on their vehicle that be oddly design and color .
  ```
- 数据格式理解及分析
   - 图片需要转换成指定大小传入。
   - 由于图片切割后和原有的groud truth未必会对应，在数据预处理时不考虑对图片进行切割。
   - 对caption需要进行word embedding预处理，并转换成对应的int类型的值，作为模型的输入。
   - 发现部分caption结尾没有句号'.'，在对数据进行预处理时可以考虑加上。
   - lemmatized后数据的有效单词数为2471，原始数据的有效单词为3081。考虑到word embedding已经起到了一定的降维和相似词整合的功能，lemmatized的作用应该不会太明显。且有效单词数量差并不明显(处于一个数量级内)。因此考虑使用原始caption作为输入数据处理。
   - Annotations.txt的作用目前没有太明白，应该类似于用户对结果的反馈评分机制。但如何运用到模型中去训练还没想清楚。

##### coco
- coco数据集目前尚未使用。

### 资料查阅
- 1411.4555论文
     - 采用CNN+RNN串联的方式，将CNN的输出作为RNN的第一个输入值
   ![embedding](pic/NIC.png)

   - word embedding大小为512
   - 要求对描述语句进行'tokenization'的预处理，我们理解为仅仅只是断词，并不需要lemmatized。但需要选择词频大于5的单词参与训练。
   - cnn有预训练模型，rnn参数从0初始化。为防止cnn预训练模型被带偏，在初始训练时需要固定cnn参数，仅训练rnn。
   后期再一起参与训练。
   - 论文中没有提到具体cnn采用了哪种模型，现有的成熟模型都可以纳入考虑。

### 项目方案规划
#### 模型规划
- CNN模型部分
  - 现有成熟的CNN网络模型较多，可供选择的也比较多。如Inception, ResNet, VGG, NasNet等
  - 仅从目前模型评价上来看，考虑到和Google的兼容性，计划采用Inception-RestNet-v2模型，20160830的预训练参数。
  - 图片输入尺寸为299*299
  - logits输出维度为[1, 1, dim_embedding]
  - 初始训练时需要锁定cnn的预训练参数。
- RNN模型部分
  - 采用LSTM长短期记忆网络模型。
  - 隐层个数为rnn_layers
  - num_steps个数根据输入caption的长度动态变化。
  - 输入及隐层的特征数与Word Embedding的维度一致，与CNN层的logit输出维度也一致。
  - 输出维度为有效单词数量，在预处理word embedding的过程中确定。使用不同数据集时会有不同的值。
  - batch size 在实际训练过程中调整。
- loss计算
  - 论文中采用的是分别计算每个词的loss并求和。但我们考虑到每句话单词数量各不相同(num_step)，会大概率导致长语句的loss偏大。会采用loss平均值来计算。

#### 系统规划
- 系统架构
  - 可运行的系统采用BS架构。
- 输入输出
  - 前端提供图片本地路径选择或URL作为输入。
  - 后端根据输入的图片，调用训练好的模型参数进行计算并输出结果语句。
- 结果展示或分析
  - 预测结果可视化
  - 根据用户反馈可进一步训练模型。

____
____


_第二周_
### 训练数据处理
- 采用的训练数据集为Flickr8k
- 参考im2txt对coco数据集的处理，修改了数据训练代码
- 设置train_shards=8 val_shards=1
- ![embedding](pic/2-6.png)
- 修改_bytes_feature函数，对图片和字符串格式分开使用了不同的函数
- ![embedding](pic/2-7.png)

### 模型训练
- #### 训练过程出现的问题以及解决
  - 训练num_steps的选择。由于flickr8k训练集的大小约为coco的1/20，我们想法是mun_steps也为50000步。
  - ![embedding](pic/2-9.png)

  - ![embedding](pic/2-8.png)
  - 设置train_inception=true，继续训练10000步
  - ![embedding](pic/2-10.png)

  - 遇到了训练好的checkpoint在本地无法调用的问题。后来发现需要checkpoint文件，并修改文件中ckpt的相对路径后成功加载恢复。

### 可用系统的搭建
  - 系统可以运行起来，如图所示
    - ![embedding](pic/2-1.png)
  - 输入：界面提供了选择文件的按钮，选择用户本地的文件
    - ![embedding](pic/2-3.png)
  - 输出：分析后生成预测结果，并提供用户反馈按钮
    - ![embedding](pic/2-2.png)

### 遇到的问题及下一阶段计划
  - 实际生成caption的过程中，计算过程稍慢，需要20-30秒左右。有没有可以提速的方法？
  - 提高生成语句的准确率，方法包括:
    - 更换数据集
    - 更换cnn模型
    - 增加训练步数num_steps
    - 调用训练过模型的checkpoint
  - 对界面异常情况进行处理，如
    - 选择非图片文件
    - 未选择文件直接点击分析
    - 无法生成caption
    - 增加多张图片批处理功能
    - 可以对每句话分别评分
  - 将用户反馈的数据加入模型训练过程，并能起到一定的影响。

____
____


_第三周_
### 模型训练完成
- 词聚类后的二维关联向量图如下：
- ![embedding](pic/tsne.png)
- 基于词关联性，训练了InceptionV3， InceptionV4， InceptionResnetV2共三个模型。
- 采用的数据集: __Flickr8k__
- 三个模型的对比分析如下：

  ##### 学习率 Learning_rate
- 三个模型采用一样的学习率
- 前10w步学习率根据epoch，学习率从2开始递减
- 10w步以后固定为0.0005
- ![embedding](pic/3-learn_rate.png)

  ##### 模型训练时间 step/sec
- InceptionV3
- ![embedding](pic/3-v3-step_sec.png)
- InceptionV4
- ![embedding](pic/3-v4-step_sec.png)
- InceptionResnetV2
- ![embedding](pic/3-v2-step_sec.png)
- 从训练时间上来看
  - cnn不训练： InceptionV3 > InceptionResnetV2 > InceptionV4
  - cnn训练后： InceptionV3 > InceptionResnetV2 ≈ InceptionV4
  - 这个速度也符合对这三个模型设计上的定义：
    - InceptionV4比InceptionV3深度更深，因此在训练和计算上都花费了更多的时间。
    - IncpetionResnetV2优化了InceptionV4的训练速度，因此速度较InceptionV4较快。

  ##### batch_loss
- InceptionV3 （处于1.7-1.9的区间范围内）
- ![embedding](pic/3-v3-batch_loss.png)
- InceptionV4 （处于1.6-2.0的区间范围内）
- ![embedding](pic/3-v4-batch_loss.png)
- InceptionResnetV2 （处于1.3-1.8的区间范围内）
- ![embedding](pic/3-v2-batch_loss.png)
- 训练过程中batch_size值为16
- 从batch_loss值的大小来看：InceptionResnetV2 < InceptionV3 < InceptionV4
- 从batch_loss值的震荡范围来看： InceptionV3 < InceptionV4 < InceptionResnetV2
- 由此分析推断，整体生成效果上：InceptionResneV2会好于其他两个模型；但对于某些特定的输入，输入结果可能存在较大的差异，不如InceptionV3输出的结果稳定。

##### total_loss
- InceptionV3 （处于1.8-2.0的区间范围内）
- ![embedding](pic/3-v3-total_loss.png)
- InceptionV4 （处于2.0-2.4的区间范围内）
- ![embedding](pic/3-v4-total_loss.png)
- InceptionResnetV2 （处于1.7-2.2的区间范围内）
- ![embedding](pic/3-v2-total_loss.png)
- 整体情况与batch_loss类似。
- InceptionV3的变化范围更小
- InceptionResnetV2收敛后的loss值更小，但浮动较大。

##### 效果分析
- 由于flickr8k原本的数据集就比较小，参照词频表对效果进行了一些分析：
- __flickr8k测试集效果测试__
  - 数据集中出现最多的名字为"_dog_"，共出现了__6158__次
  - ![embedding](pic/sample1.png)
  - ![embedding](pic/sample5.png)
  - 对dog的识别相当准确，dog身上的毛发颜色识别的也不错。动作描述略有差别，InceptionResnetV2模型稍好一些。
  - ![embedding](pic/sample6.png)
  - 这张图三个模型识别的差别相当明显。InceptionV4的语句略短，且没有识别出court dribbling，衣服颜色等关键词。
  - 只有InceptionResnetV2模型识别出了dribbling这个动作。这个词在词频表里出现的次数仅5次。说明InceptionResnetV2模型对低频词的学习和使用较为频繁。另一个类似的例子是：
  - ![embedding](pic/sample3.png)
  - 这里使用petting更为准确。但InceptionResnetV2模型多次使用了tug of war 这个冷僻的词组结构。它在训练集中仅仅出现过__2__次。
- __范围外的图像效果测试__
  - ![embedding](pic/sample2.png)
  - 由于词组表中没有zebra，导致测试结果惨不忍睹。
  - 结果中出现的man/woman/boy/girl/child都是词频表中出现次数最多的一些名词（2000-5000次）
  - 唯一欣慰的是InceptionResnetV2中出现了horse这个单词。他在词频表里出现了169次。
- __网络图片效果测试__
  - ![embedding](pic/sample8.png)
  - 词库中关于雪的出现的次数比较多，网上找了一些雪景的图片测试，效果还不错。
  - ![embedding](pic/sample9.png)
  - 对于高频词的处理InceptionResnetV2表现并不太好。如上例中，对man/woman的区分并不明显，语句也并不通顺。反而InceptionV4表达的意思更准确。

### 系统搭建完成
  - 能运行并根据合理的输入给出合理的输出
    - 系统主界面
    - ![embedding](pic/3-web-1.png)
    - 生成结果图片，支持三种模型的结果的同时预测
    - ![embedding](pic/sample1.png)
  - 一些异常和故障处理
    - 选择文件对话框对文件类型做了筛选，只能选择图片类型的文件。如jpg/jpeg/png
    - ![embedding](pic/3-web-2.png)
    - 对文件的最小尺寸有限制，尺寸过小会弹出提示
    - ![embedding](pic/3-web-3.png)
    - 直接点击提交按钮有错误提示
    - ![embedding](pic/3-web-5.png)
    - 计算过程中重复提交图片，会有不要重复点击提示
    - ![embedding](pic/3-web-4.png)


____
____

_第四周_
- 项目各个阶段中的坑
  - 数据建模过程
    - flickr8k的数据集和mscoco的数据结构差别较大。mscoco采用json封装了caption的内容，而flickr8k的数据集则是简单的采用txt格式的文本。并把训练集和验证集的caption放在了一起。
    - 在借用mscoco的tfrecord代码的时，由于编码方式的错误，导致生成了空的数据。
  - 数据训练过程
    - 由于模型的参数没有设置正确（如词组数量），导致在验证过程时才发现输出的词组维度与词汇表不符，之前的训练过程全部作废。
    - 由于代码中设置了learningrate按照epoch逐渐递减的策略，但epoch设置过大，导致学习率无法递减，模型一直跑在了一个极高的学习率上。
  - 验证生成过程
    - 之前由于没有了解beamsearch搜索算法，导致生成的语句不够合理。以为是训练量不够的问题，试图尝试加大训练量来完成。结果当然无济于事。
    - 尝试用BELU在验证集上比较生成的caption和原始caption的差异。
  - 系统搭建过程
    - 由于没有在加载网页的时候就构建模型，导致计算caption的时间特别的长。修改后端逻辑后，同时计算三个模型的caption也没有花费太多的时间。
- 心得体会
  - 朱怡心得
  ````
  整个项目的时间比较紧，大部分工作都是在下班后和周末匆匆忙忙赶出来的。
  开始的时候考虑了很多，但最终在很多细节上也都做得比较粗糙。
  很多知识点，比如beam search/BELU等也都没有仔细的推敲研究，只是或囵吞枣的拿来用了。
  所幸项目最终的展现效果自我感觉还是不错的。
  这是第一次完整的做一个深度学习方面的项目，在之前课程的支持下，进行的倒也还算顺利。
  反倒是为了做个网页，现学了h5/js/django，花了不少的时间。
  比较遗憾的是由于设备所限，最终没有能在mscoco训练集上训练出一个效果较好的模型。
  ````
  - 张臣生心得
  ````
  1，通过项目体会到了cnn与LSTM结合完成项目的神奇效果。在项目之前以为句子的语法会是一大难题，但是实际下来反而是句法很少出问题。
  2，通过项目学习到了cnn提取的特征并不仅限于名词，在句子中涉及到了相当多的动词，多物体的相对位置等信息，cnn也可以很好的提取信息并由LSTM得到最后的结果。
  3，通过项目也认识到计算资源与数据样本在大数据中的重要性。没有大的样本，经常会出现过拟合现象，名词部分也经常出错。
  4，但是在项目中对于cnn与LSTM都使用了框架，内部的计算还没有完全搞的很透彻。由于时间较短，也没有采取更多的数据处理与分析。
  5，项目过程中遇到相当多的工程问题，而并不是理论也计算的问题。多实践多做项目是十分需要的。
  6，在项目过程中有相当多的时候在改数据打包tfrecord，配置服务器，调试inference代码。而对于模型核心部分的代码因为使用了slim框架反而没有消耗太多时间。
  7，项目过程中团队配合非常重要，特别感谢同组的组员对于项目的奉献。
  ````
  - 李雨阳心得
  ````
  整个项目的时间比较紧，由于近期由于工作变动的原因所以参与度相比于其他组员较少，在这里先感谢怡神和臣生大神对项目的贡献。
  刚开始做项目的时候想的很多，想过对ground truth进行解析然后进行语句分析，例如主谓宾等，但经过一些论文的研究发现这种方法并不可取，反而有可能丧失神经网络自动提取特征的优势。
  项目过程中使用的大多都是框架，大多处理的都是工程上的一些问题，即使如此还是遇到了很多问题，很多想起来很简单的过程，实际处理过程中会有很多意想不到的问题发生，对于接下来的学习，更应该踏实的学，踏实的动手去做，才能有更进一步的发展。
  项目过程中去了解了一下基础的爬虫技术，爬了一下有道辞典，实现了一个小的翻译功能，感觉还是非常有意思的，但同时也感觉这是一个非常吃经验的行业。
  这次项目大多都是使用现有的框架，对内部的一些细节了解不是很深，希望自己今后能在工程实践的基础上更进一步，详细研究一下中间的过程，结合最新研究成果和实用背景，从实践中提取理论思想。
  ````
- 项目不足与改进设想
  - 对生成结果的评价不足，由于没有搞明白，所以没好意思在输出的caption上加上置信度。
  - 由于使用了flickr8k数据集，使用的模型规模太小。总共3000个词组的模型存在着严重的过拟合情况，没有进行有效的规避。如果采用mscoco的数据集进行训练，效果应该会好很多。
  - 原来设想的用户评价，反馈训练的功能没能够实现。
  - 语言上目前仅支持英语生成，依靠外部翻译软件实现语句的中文翻译。如果有中文生成的话会显得更完善。
  - 界面虽然元素齐全，但还是显得较为粗糙。
  - 从系统扩展性上，目前系统界面上仅支持生成图片标题。如果可以在界面上直接通过导入数据来训练特有的新模型，系统功能会更为完善。可应用于某些特定场景监视或监控功能。

- 项目安装说明
  - 数据格式生成环境配置
    - 生成mscoco tfrecord数据的脚本为 /im2txt/data/build_mscoco_data.py
    - 命令参数说明如下
    ````
    python build_mscoco_data.py \
    --train_image_dir="./train2014" \                   ## 训练集图片目录
    --val_image_dir="./val2014" \                       ## 验证集图片目录
    --train_captions_file="./captions_train2014.json" \ ## 训练集标签所在路径
    --val_captions_file="./captions_val2014.json" \     ## 验证集标签所在路径
    --output_dir="./outputs" \                          ## 输出文件目录
    --word_counts_output_file="./word_counts.txt"       ## 输出词频文件路径
    ````
    - 生成flickr8k tfrecord数据的脚本为 /im2txt/data/build_flickr8k_data.py
    ````
    python build_flickr8k_data.py \
    --train_image_dir="./images" \           ## 图片集目录
    --val_image_dir="./images" \             ## 图片集目录
    --captions_file="./Flickr8k.token.txt" \ ## 图片标签所在路径
    --output_dir="./outputs" \               ## 输出文件目录
    --word_counts_output_file="./word_counts.txt"       ## 输出词频文件路径
    ````
  - 模型训练环境配置
    - 模型训练的脚本为 /im2txt/train.py
    - 命令及参数说明如下
    ````
    train.py \
    --input_file_pattern='./train-?????-of-00008' \  ## 按照某个格式的tfrecord文件
    --inception_checkpoint_file='inception_v3.ckpt'\ ## 模型的ckpt，需要与cnn_model匹配
    --train_dir='./output' \                     ## 训练模型的输出目录
    --train_inception='false' \                  ## 是否训练cnn模型，建议一开始为false，loss不再下降后改为true继续训练
    --number_of_steps='100000' \                 ## 总训练步数
    --log_every_n_steps='1' \                    ## log输出的步数
    --cnn_model='InceptionV3'                    ## 调用的cnn模型，目前仅支持'InceptionV3', 'InceptionV4', 'InceptionResnetV2' 三种模型。
    ````
    -
    - configuration.py中的部分参数需要根据项目情况修改，如
    `````
    self.values_per_input_shard = 3750           ## 每个shard的训练数目
    self.vocab_size = 3000                       ## caption词汇表总数量
    self.batch_size = 16
    self.num_examples_per_epoch = 30000          ## 训练集数量        
    `````
  - web界面及后端环境配置
    - web后端的工程地址为 /im2txt/webserver/myweb
    - 本地调试命令
    ````
    python manage.py runserver
    ````
    - 也可采用Nginx + uwsgi 或 apache 部署django

- 项目使用说明
  - 相关地址
  ````
    - 项目gitee地址：https://gitee.com/itiszogo/image-caption
    - 项目github地址：https://github.com/zhuyi55/img_caption
    - 项目训练模型下载地址：https://pan.baidu.com/s/1YaBSxlESy5ez1zbqnzBRrw 提取码:fmsc
    - 也可以通过上述代码自行修改训练模型
  ````
  - 训练模型需要放到 /web_server/myweb/im_model 路径下
  - 在本地可通过以下命令启动服务器
  ````
  python manage.py runserver
  ````
  - 在浏览器中如下输入，启动客户端界面
  ````
  http://localhost:8000/upload
  或其他项目环境配置的ip地址和端口
  ````
  - 通过一些时间的启动加载后，网页上应显示如下内容（启动加载时间取决于服务器端模型加载的速度）
  - ![embedding](pic/3-web-1.png)
    - 选择文件按钮可以选择本地的一张图片进行看图说话的操作。建议选择图片尺寸大于299×299。
    - 三个勾选框对应三个不同的看图说话模型，建议全勾选以获得最好的生成效果。每种模型将生成3条语句。三个模型共9条。
    - 分析按钮会提交当前选择的图片信息，对该图片进行计算。计算过程中请勿重复提交。
  - 服务器计算完成后，会显示如下信息
  - ![embedding](pic/4-sample.png)

- 使用展示视频
  - 参见document/lookme.mp4
