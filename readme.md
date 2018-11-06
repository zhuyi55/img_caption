## 看图说话机器人项目说明
- github地址：https://github.com/zhuyi55/img_caption
- flickr8k数据集地址：https://www.tinymind.com/zhuyi55/datasets/flickr8k
- 其他

### 代码模块说明
````
 + documents
   - 1411.4555.pdf                                    <相关论文>
 - convert_flickr8k_dataset.py                       <数据集tfrecord打包模块>
 - dataset.py	train                                  <数据处理模块>
 - dictionary.json                                   <单词序号映射字典>
 - embedding.npy                                     <Word Embedding 矩阵>
 - flags.py                                          <可调参数定义模块>
 - reverse_dictionary.json                           <反向映射字典>
 - train.py                                          <训练模块>
 - tsne.png                                          <Word Embedding 矩阵图>
 - vgg.py                                            <vgg模型代码>
 - word2vec.py                                       <Word Embedding 映射模块>
````
##### Word Embedding (word2vec.py)
- 对于新的数据集首先进行Word embedding处理。
- 按照论文中的做法，仅处理词频大于5的单词。代码中调用 build_dataset(words, n_words)，将n_words设置的略大，函数可以自动收敛到词频至少为5的单词个数。 如flickr8k数据集的单词数为3108
- 新数据集的nwords需设置到flag.py的参数配置文件中，'num_words' 参数，或在训练模型时修改。

##### tfrecord (convert_flickr8k_dataset.py)
- 总共会生成4个tfrecord文件
````
 - flickr8k_train_one.record:     单张图片5个sentense
 - flickr8k_train.record:         6000张图片， 30000条数据
 - flickr8k_val.record:           1000张图片， 5000条数据
 - flickr8k_test.record:          1000张图片， 5000条数据
````
- 对输入的图片没做处理，默认输入为jpg图片。对其他格式的图片未作处理。
- 对输入的语句进行了处理
 - 调用dictionary.json对单词进行了转换，转换成int数组。
 - 由于每句话长短不一，对数据进行了长度的统一为64。（方便建模，后续可以考虑改为动态大小）

##### dataset module (dataset.py)
- tfrecord解包过程
- 考虑到切割后的图片并没有意义，因此没有进行图像切割的操作，而是直接对图像进行了resize
- 目前代码中使用了vgg16的cnn模型，resize尺寸对应的为[224, 224]

##### train module (train.py)
- 主要的建模和训练过程
- 模型后续需要封装以增强代码的可读性
- 目前使用了最简单的 vgg16 + 3层lstm结构
