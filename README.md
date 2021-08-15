
# [AI训练营]PaddleX实现目标检测baseline

手把手教你基于PaddleX实现目标检测。你需要实现以下任务：

> 1. 配置数据集（数据集选择、数据处理）
> 2. 配置模型并训练
> 3. 项目跑通即可达到结业要求

# 一、数据集说明

本项目使用的数据集是：[[AI训练营]目标检测数据集合集](https://aistudio.baidu.com/aistudio/datasetdetail/103743)，包含口罩识别 、交通标志识别、火焰检测、锥桶识别以及中秋元素识别。

该数据集已加载至本环境中，位于：**data/data103743/objDataset.zip**


```python
# 解压数据集（解压一次即可，请勿重复解压）
!unzip -oq /home/aistudio/data/data103743/objDataset.zip
```

解压完成后，左侧文件夹处会多一个名为**objDataset**的文件夹，该文件夹下有5个子文件夹：
- **barricade**——Gazebo锥桶检测
- **facemask**——口罩检测
- **fire**——火焰检测
- **MidAutumn**——中秋元素检测
- **roadsign_voc**——交通路标检测

每个子文件夹下有2个文件夹，分别存放着图像（**JPEGImages**）和标注文件（**Annotations**），如下所示：


```python
# 查看数据集文件结构
!tree objDataset -L 2
```

    objDataset
    ├── barricade
    │   ├── Annotations
    │   └── JPEGImages
    ├── facemask
    │   ├── Annotations
    │   └── JPEGImages
    ├── fire
    │   ├── Annotations
    │   └── JPEGImages
    ├── MidAutumn
    │   ├── Annotations
    │   └── JPEGImages
    └── roadsign_voc
        ├── Annotations
        └── JPEGImages
    
    15 directories, 0 files


# 二、数据准备

本基线系统使用的数据格式是PascalVOC格式，开发者基于PaddleX开发目标检测模型时，无需对数据格式进行转换，开箱即用。

但为了进行训练，还需要将数据划分为训练集、验证集和测试集。划分之前首先需要**安装PaddleX**。


```python
# 安装PaddleX
!pip install paddlex
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting paddlex
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d6/a2/07435f4aa1e51fe22bdf06c95d03bf1b78b7bc6625adbb51e35dc0804cc7/paddlex-1.3.11-py3-none-any.whl (516kB)
    [K     |████████████████████████████████| 522kB 12.0MB/s eta 0:00:01
    [?25hRequirement already satisfied: flask-cors in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.8)
    Collecting paddleslim==1.1.1 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d1/77/e257227bed9a70ff0d35a4a3c4e70ac2d2362c803834c4c52018f7c4b762/paddleslim-1.1.1-py2.py3-none-any.whl (145kB)
    [K     |████████████████████████████████| 153kB 30.3MB/s eta 0:00:01
    [?25hRequirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.2.0)
    Requirement already satisfied: psutil in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.7.2)
    Collecting xlwt (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/44/48/def306413b25c3d01753603b1a222a011b8621aed27cd7f89cbc27e6b0f4/xlwt-1.3.0-py2.py3-none-any.whl (99kB)
    [K     |████████████████████████████████| 102kB 33.0MB/s ta 0:00:01
    [?25hRequirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.1.1.26)
    Requirement already satisfied: sklearn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.0)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.36.1)
    Requirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.4.4)
    Collecting shapely>=1.7.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/98/f8/db4d3426a1aba9d5dfcc83ed5a3e2935d2b1deb73d350642931791a61c37/Shapely-1.7.1-cp37-cp37m-manylinux1_x86_64.whl (1.0MB)
    [K     |████████████████████████████████| 1.0MB 14.3MB/s eta 0:00:01
    [?25hCollecting paddlehub==2.1.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/7a/29/3bd0ca43c787181e9c22fe44b944b64d7fcb14ce66d3bf4602d9ad2ac76c/paddlehub-2.1.0-py3-none-any.whl (211kB)
    [K     |████████████████████████████████| 215kB 26.0MB/s eta 0:00:01
    [?25hRequirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.1.2)
    Collecting pycocotools; platform_system != "Windows" (from paddlex)
      Downloading https://mirror.baidu.com/pypi/packages/de/df/056875d697c45182ed6d2ae21f62015896fdb841906fe48e7268e791c467/pycocotools-2.0.2.tar.gz
    Requirement already satisfied: Six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask-cors->paddlex) (1.15.0)
    Requirement already satisfied: Flask>=0.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask-cors->paddlex) (1.1.1)
    Requirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddleslim==1.1.1->paddlex) (18.1.1)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.21.0)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.2.3)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.1.5)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.8.2)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.0.0)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (7.1.2)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.14.0)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.20.3)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.22.0)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.8.53)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.7.1.1)
    Requirement already satisfied: scikit-learn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from sklearn->paddlex) (0.24.2)
    Requirement already satisfied: filelock in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.0.12)
    Requirement already satisfied: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (4.1.0)
    Requirement already satisfied: packaging in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.9)
    Collecting paddle2onnx>=0.5.1 (from paddlehub==2.1.0->paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/37/80/aa6134b5f36aea45dc1b363e7af941dccabe4d7e167ac391ff046f34baf1/paddle2onnx-0.7-py3-none-any.whl (94kB)
    [K     |████████████████████████████████| 102kB 24.2MB/s ta 0:00:01
    [?25hRequirement already satisfied: rarfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1)
    Requirement already satisfied: paddlenlp>=2.0.0rc5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.0.7)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.9)
    Requirement already satisfied: gunicorn>=19.10.0; sys_platform != "win32" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.0.4)
    Requirement already satisfied: gitpython in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1.14)
    Requirement already satisfied: setuptools>=18.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (56.2.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (0.29)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (7.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (2.10.1)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (1.1.0)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (0.16.0)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (16.7.9)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.4)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.23)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (2.0.1)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.4.10)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (2.4.2)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (2019.3)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (1.1.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.2.0)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.6.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.1)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (1.25.6)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2.8)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2019.9.11)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (3.0.4)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (3.9.9)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (0.18.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (2.1.0)
    Requirement already satisfied: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (1.6.3)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (0.14.1)
    Requirement already satisfied: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.70.11.1)
    Requirement already satisfied: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (2.9.0)
    Requirement already satisfied: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (1.2.2)
    Requirement already satisfied: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.42.1)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitpython->paddlehub==2.1.0->paddlex) (4.0.5)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->Flask>=0.9->flask-cors->paddlex) (1.1.1)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0->paddlex) (0.6.0)
    Requirement already satisfied: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.3.3)
    Requirement already satisfied: smmap<4,>=3.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython->paddlehub==2.1.0->paddlex) (3.0.5)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0->paddlex) (7.2.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (setup.py) ... [?25ldone
    [?25h  Created wheel for pycocotools: filename=pycocotools-2.0.2-cp37-cp37m-linux_x86_64.whl size=278367 sha256=9d6ca8785b510be3c060f8d00687b7fc1ede836602e23b960a1eb6c1b71ce0cb
      Stored in directory: /home/aistudio/.cache/pip/wheels/fb/44/67/8baa69040569b1edbd7776ec6f82c387663e724908aaa60963
    Successfully built pycocotools
    Installing collected packages: paddleslim, xlwt, shapely, paddle2onnx, paddlehub, pycocotools, paddlex
      Found existing installation: paddlehub 2.0.4
        Uninstalling paddlehub-2.0.4:
          Successfully uninstalled paddlehub-2.0.4
    Successfully installed paddle2onnx-0.7 paddlehub-2.1.0 paddleslim-1.1.1 paddlex-1.3.11 pycocotools-2.0.2 shapely-1.7.1 xlwt-1.3.0


使用如下命令即可将数据划分为70%训练集，20%验证集和10%的测试集。


```python
# 划分数据集
!paddlex --split_dataset --format VOC --dataset_dir objDataset/barricade --val_value 0.2 --test_value 0.1
```

    Dataset Split Done.[0m
    [0mTrain samples: 364[0m
    [0mEval samples: 104[0m
    [0mTest samples: 52[0m
    [0mSplit files saved in objDataset/barricade[0m
    [0m[0m

划分完成后，该数据集下会生成**labels.txt**, **train_list.txt**, **val_list.txt**和**test_list.txt**，分别存储类别信息，训练样本列表，验证样本列表，测试样本列表。如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/d29a92b4cfc34b0097ef46dbbc8562af824387889f224948ae49283e0adee19d)

在这里，**你需要将path to dataset部分替换成你选择的数据集路径**。在左侧文件夹处，将鼠标放到你想选择的数据集文件夹上，会出现三个图标，第一个图标表示复制该文件夹路径，点击即可获得该文件夹路径，用这个路径替换path to dataset即可。

![](https://ai-studio-static-online.cdn.bcebos.com/c28ed88586644f64b34709a592fea0b97ec80470c0e041fd9aa6b8da21c8e283)


# 三、数据预处理

在训练模型之前，对目标检测任务的数据进行操作，从而提升模型效果。可用于数据处理的API有：
- **Normalize**：对图像进行归一化
- **ResizeByShort**：根据图像的短边调整图像大小
- **RandomHorizontalFlip**：以一定的概率对图像进行随机水平翻转
- **RandomDistort**：以一定的概率对图像进行随机像素内容变换

更多关于数据处理的API及使用说明可查看文档：
[https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html](https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html)


```python
from paddlex.det import transforms

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], min_val=[0., 0., 0.], max_val=[255., 255., 255.]),
    transforms.RandomHorizontalFlip(prob=0.5),
    transforms.RandomDistort(brightness_range=0.5, brightness_prob=0.5, contrast_range=0.5, contrast_prob=0.5, saturation_range=0.5, saturation_prob=0.5, hue_range=18, hue_prob=0.5),
    transforms.ResizeByShort(short_size=800, max_size=1333),
])

eval_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], min_val=[0., 0., 0.], max_val=[255., 255., 255.]),
    transforms.RandomHorizontalFlip(prob=0.5),
    transforms.RandomDistort(brightness_range=0.5, brightness_prob=0.5, contrast_range=0.5, contrast_prob=0.5, saturation_range=0.5, saturation_prob=0.5, hue_range=18, hue_prob=0.5),
    transforms.ResizeByShort(short_size=800, max_size=1333),
])
```

读取PascalVOC格式的检测数据集，并对样本进行相应的处理。


```python
import paddlex as pdx

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/barricade',
    file_list='objDataset/barricade/train_list.txt',
    label_list='objDataset/barricade/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/barricade',
    file_list='objDataset/barricade/val_list.txt',
    label_list='objDataset/barricade/labels.txt',
    transforms=eval_transforms)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


    2021-08-15 12:20:55 [INFO]	Starting to read file list from dataset...
    2021-08-15 12:20:55 [INFO]	364 samples in file objDataset/barricade/train_list.txt
    creating index...
    index created!
    2021-08-15 12:20:55 [INFO]	Starting to read file list from dataset...
    2021-08-15 12:20:55 [INFO]	104 samples in file objDataset/barricade/val_list.txt
    creating index...
    index created!


需要注意的是：
- **data_dir** (str): 数据集所在的目录路径。
- **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路径）。
- **label_list** (str): 描述数据集包含的类别信息文件路径。

需要将第二步数据准备时生成的labels.txt, train_list.txt, val_list.txt和test_list.txt配置到以上变量中，如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/6462f7811da3436290948e5dde0c497d6ae51bcfe1904e0a863ff032363a4448)


# 四、模型训练

PaddleX目前提供了FasterRCNN和YOLOv3两种检测结构，多种backbone模型。本基线系统以骨干网络为MobileNetV1的YOLOv3算法为例。


```python
# 初始化模型
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3

# 此处需要补充目标检测模型代码
#model = pdx.det.YOLOv3(num_classes=len(train_dataset.labels), backbone='MobileNetV1')
model = pdx.det.PPYOLO(num_classes=80, backbone='ResNet50_vd_ssld', with_dcn_v2=True, anchors=None, anchor_masks=None, use_coord_conv=True, use_iou_aware=True, use_spp=True, use_drop_block=True, scale_x_y=1.05, ignore_threshold=0.7, label_smooth=False, use_iou_loss=True, use_matrix_nms=True, nms_score_threshold=0.01, nms_topk=1000, nms_keep_topk=100, nms_iou_threshold=0.45, train_random_shapes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608], input_channel=3)
```


```python
# 模型训练
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html

# 此处需要补充模型训练参数
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    lr_decay_epochs=[210, 240],
    lr_decay_gamma=0.1,
    save_dir='output/yolov3_mobilenetv1')
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py:706: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      elif dtype == np.bool:
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py:2043: UserWarning: The Attr(force_cpu) of Op(fill_constant) will be deprecated in the future, please use 'device_guard' instead. 'device_guard' has higher priority when they are used at the same time.
      "used at the same time." % type)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/ops.py:131
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:155
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:172
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:172
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:174
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:174
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:178
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:178
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:180
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:180
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:216
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:217
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:218
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:219
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:97
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:97
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:99
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:101
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:101
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:101
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:102
    The behavior of expression A / B has been unified with elementwise_div(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_div(X, Y, axis=0) instead of A / B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/iou_loss.py:79
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:186
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:194
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:349
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:350
    The behavior of expression A - B has been unified with elementwise_sub(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_sub(X, Y, axis=0) instead of A - B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:351
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:352
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:383
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:385
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:209
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:210
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/loss/yolo_loss.py:212
    The behavior of expression A + B has been unified with elementwise_add(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_add(X, Y, axis=0) instead of A + B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/iou_aware.py:64
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:322: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/iou_aware.py:40
    The behavior of expression A / B has been unified with elementwise_div(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_div(X, Y, axis=0) instead of A / B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))


    2021-08-15 12:26:56 [INFO]	Downloading ResNet50_vd_ssld_pretrained.tar from https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar


    100%|██████████| 92837/92837 [00:01<00:00, 55915.48KB/s]


    2021-08-15 12:26:58 [INFO]	Decompressing output/yolov3_mobilenetv1/pretrain/ResNet50_vd_ssld_pretrained.tar...
    2021-08-15 12:27:05 [INFO]	Load pretrain weights from output/yolov3_mobilenetv1/pretrain/ResNet50_vd_ssld_pretrained.
    2021-08-15 12:27:05 [INFO]	There are 275 varaibles in output/yolov3_mobilenetv1/pretrain/ResNet50_vd_ssld_pretrained are loaded.
    2021-08-15 12:27:18 [INFO]	[TRAIN] Epoch=1/270, Step=2/45, loss=13779.546875, lr=0.0, time_each_step=6.13s, eta=20:59:58
    2021-08-15 12:27:19 [INFO]	[TRAIN] Epoch=1/270, Step=4/45, loss=7534.054688, lr=0.0, time_each_step=3.35s, eta=11:27:24
    2021-08-15 12:27:20 [INFO]	[TRAIN] Epoch=1/270, Step=6/45, loss=4695.144531, lr=1e-06, time_each_step=2.43s, eta=8:18:36
    2021-08-15 12:27:21 [INFO]	[TRAIN] Epoch=1/270, Step=8/45, loss=13434.888672, lr=1e-06, time_each_step=2.0s, eta=6:50:22
    2021-08-15 12:27:23 [INFO]	[TRAIN] Epoch=1/270, Step=10/45, loss=2592.106934, lr=1e-06, time_each_step=1.71s, eta=5:51:0
    2021-08-15 12:27:24 [INFO]	[TRAIN] Epoch=1/270, Step=12/45, loss=2298.955566, lr=1e-06, time_each_step=1.53s, eta=5:14:5
    2021-08-15 12:27:25 [INFO]	[TRAIN] Epoch=1/270, Step=14/45, loss=1508.481934, lr=2e-06, time_each_step=1.39s, eta=4:45:13
    2021-08-15 12:27:26 [INFO]	[TRAIN] Epoch=1/270, Step=16/45, loss=714.810974, lr=2e-06, time_each_step=1.3s, eta=4:26:18
    2021-08-15 12:27:28 [INFO]	[TRAIN] Epoch=1/270, Step=18/45, loss=610.801025, lr=2e-06, time_each_step=1.23s, eta=4:11:29
    2021-08-15 12:27:29 [INFO]	[TRAIN] Epoch=1/270, Step=20/45, loss=598.355103, lr=2e-06, time_each_step=1.17s, eta=3:59:50
    2021-08-15 12:27:30 [INFO]	[TRAIN] Epoch=1/270, Step=22/45, loss=297.958252, lr=3e-06, time_each_step=0.62s, eta=2:7:18
    2021-08-15 12:27:31 [INFO]	[TRAIN] Epoch=1/270, Step=24/45, loss=321.912415, lr=3e-06, time_each_step=0.63s, eta=2:8:57
    2021-08-15 12:27:33 [INFO]	[TRAIN] Epoch=1/270, Step=26/45, loss=253.317551, lr=3e-06, time_each_step=0.64s, eta=2:11:11
    2021-08-15 12:27:34 [INFO]	[TRAIN] Epoch=1/270, Step=28/45, loss=197.047089, lr=3e-06, time_each_step=0.63s, eta=2:8:51
    2021-08-15 12:27:35 [INFO]	[TRAIN] Epoch=1/270, Step=30/45, loss=153.639435, lr=4e-06, time_each_step=0.64s, eta=2:10:53
    2021-08-15 12:27:37 [INFO]	[TRAIN] Epoch=1/270, Step=32/45, loss=176.438263, lr=4e-06, time_each_step=0.65s, eta=2:12:20
    2021-08-15 12:27:37 [INFO]	[TRAIN] Epoch=1/270, Step=34/45, loss=153.89682, lr=4e-06, time_each_step=0.62s, eta=2:7:33
    2021-08-15 12:27:38 [INFO]	[TRAIN] Epoch=1/270, Step=36/45, loss=137.25444, lr=4e-06, time_each_step=0.58s, eta=1:57:56
    2021-08-15 12:27:39 [INFO]	[TRAIN] Epoch=1/270, Step=38/45, loss=188.097717, lr=5e-06, time_each_step=0.55s, eta=1:52:50
    2021-08-15 12:27:39 [INFO]	[TRAIN] Epoch=1/270, Step=40/45, loss=110.468285, lr=5e-06, time_each_step=0.51s, eta=1:44:39
    2021-08-15 12:27:40 [INFO]	[TRAIN] Epoch=1/270, Step=42/45, loss=151.280273, lr=5e-06, time_each_step=0.47s, eta=1:36:22
    2021-08-15 12:27:40 [INFO]	[TRAIN] Epoch=1/270, Step=44/45, loss=132.274323, lr=5e-06, time_each_step=0.43s, eta=1:27:31
    2021-08-15 12:27:40 [INFO]	[TRAIN] Epoch 1 finished, loss=2676.27002, lr=3e-06 .
    2021-08-15 12:27:46 [INFO]	[TRAIN] Epoch=2/270, Step=1/45, loss=127.800858, lr=6e-06, time_each_step=0.67s, eta=2:37:28
    2021-08-15 12:27:48 [INFO]	[TRAIN] Epoch=2/270, Step=3/45, loss=141.4151, lr=6e-06, time_each_step=0.71s, eta=2:37:35
    2021-08-15 12:27:50 [INFO]	[TRAIN] Epoch=2/270, Step=5/45, loss=103.121246, lr=6e-06, time_each_step=0.72s, eta=2:37:35
    2021-08-15 12:27:51 [INFO]	[TRAIN] Epoch=2/270, Step=7/45, loss=134.666183, lr=6e-06, time_each_step=0.73s, eta=2:37:37
    2021-08-15 12:27:53 [INFO]	[TRAIN] Epoch=2/270, Step=9/45, loss=95.991859, lr=7e-06, time_each_step=0.8s, eta=2:37:50
    2021-08-15 12:27:55 [INFO]	[TRAIN] Epoch=2/270, Step=11/45, loss=104.402573, lr=7e-06, time_each_step=0.88s, eta=2:38:4
    2021-08-15 12:27:56 [INFO]	[TRAIN] Epoch=2/270, Step=13/45, loss=137.027466, lr=7e-06, time_each_step=0.89s, eta=2:38:7
    2021-08-15 12:27:58 [INFO]	[TRAIN] Epoch=2/270, Step=15/45, loss=145.912292, lr=7e-06, time_each_step=0.94s, eta=2:38:15
    2021-08-15 12:27:59 [INFO]	[TRAIN] Epoch=2/270, Step=17/45, loss=115.586998, lr=8e-06, time_each_step=0.96s, eta=2:38:18
    2021-08-15 12:28:00 [INFO]	[TRAIN] Epoch=2/270, Step=19/45, loss=141.337067, lr=8e-06, time_each_step=1.01s, eta=2:38:26
    2021-08-15 12:28:01 [INFO]	[TRAIN] Epoch=2/270, Step=21/45, loss=122.781319, lr=8e-06, time_each_step=0.75s, eta=2:37:30
    2021-08-15 12:28:03 [INFO]	[TRAIN] Epoch=2/270, Step=23/45, loss=111.738808, lr=8e-06, time_each_step=0.71s, eta=2:37:21
    2021-08-15 12:28:04 [INFO]	[TRAIN] Epoch=2/270, Step=25/45, loss=123.957672, lr=9e-06, time_each_step=0.7s, eta=2:37:18
    2021-08-15 12:28:05 [INFO]	[TRAIN] Epoch=2/270, Step=27/45, loss=116.265846, lr=9e-06, time_each_step=0.68s, eta=2:37:11
    2021-08-15 12:28:06 [INFO]	[TRAIN] Epoch=2/270, Step=29/45, loss=125.722214, lr=9e-06, time_each_step=0.62s, eta=2:36:59
    2021-08-15 12:28:07 [INFO]	[TRAIN] Epoch=2/270, Step=31/45, loss=113.825089, lr=9e-06, time_each_step=0.58s, eta=2:36:50
    2021-08-15 12:28:08 [INFO]	[TRAIN] Epoch=2/270, Step=33/45, loss=90.576759, lr=1e-05, time_each_step=0.57s, eta=2:36:46
    2021-08-15 12:28:08 [INFO]	[TRAIN] Epoch=2/270, Step=35/45, loss=118.262527, lr=1e-05, time_each_step=0.52s, eta=2:36:35
    2021-08-15 12:28:09 [INFO]	[TRAIN] Epoch=2/270, Step=37/45, loss=136.333389, lr=1e-05, time_each_step=0.49s, eta=2:36:29
    2021-08-15 12:28:09 [INFO]	[TRAIN] Epoch=2/270, Step=39/45, loss=122.885628, lr=1e-05, time_each_step=0.45s, eta=2:36:20
    2021-08-15 12:28:21 [INFO]	[TRAIN] Epoch=3/270, Step=2/45, loss=106.166443, lr=1.1e-05, time_each_step=0.79s, eta=2:19:19
    2021-08-15 12:28:22 [INFO]	[TRAIN] Epoch=3/270, Step=4/45, loss=104.364471, lr=1.2e-05, time_each_step=0.83s, eta=2:19:26
    2021-08-15 12:28:24 [INFO]	[TRAIN] Epoch=3/270, Step=6/45, loss=113.172028, lr=1.2e-05, time_each_step=0.86s, eta=2:19:32
    2021-08-15 12:28:26 [INFO]	[TRAIN] Epoch=3/270, Step=8/45, loss=116.821159, lr=1.2e-05, time_each_step=0.91s, eta=2:19:41
    2021-08-15 12:28:28 [INFO]	[TRAIN] Epoch=3/270, Step=10/45, loss=95.0998, lr=1.2e-05, time_each_step=0.97s, eta=2:19:52
    2021-08-15 12:28:29 [INFO]	[TRAIN] Epoch=3/270, Step=12/45, loss=107.954926, lr=1.3e-05, time_each_step=1.0s, eta=2:19:57
    2021-08-15 12:28:30 [INFO]	[TRAIN] Epoch=3/270, Step=14/45, loss=103.331451, lr=1.3e-05, time_each_step=1.05s, eta=2:20:5
    2021-08-15 12:28:32 [INFO]	[TRAIN] Epoch=3/270, Step=16/45, loss=110.119934, lr=1.3e-05, time_each_step=1.1s, eta=2:20:15
    2021-08-15 12:28:33 [INFO]	[TRAIN] Epoch=3/270, Step=18/45, loss=89.324234, lr=1.3e-05, time_each_step=1.14s, eta=2:20:21
    2021-08-15 12:28:35 [INFO]	[TRAIN] Epoch=3/270, Step=20/45, loss=88.216103, lr=1.4e-05, time_each_step=1.19s, eta=2:20:28
    2021-08-15 12:28:36 [INFO]	[TRAIN] Epoch=3/270, Step=22/45, loss=109.850227, lr=1.4e-05, time_each_step=0.75s, eta=2:18:55
    2021-08-15 12:28:37 [INFO]	[TRAIN] Epoch=3/270, Step=24/45, loss=87.792587, lr=1.4e-05, time_each_step=0.72s, eta=2:18:47
    2021-08-15 12:28:38 [INFO]	[TRAIN] Epoch=3/270, Step=26/45, loss=113.469078, lr=1.4e-05, time_each_step=0.7s, eta=2:18:43
    2021-08-15 12:28:39 [INFO]	[TRAIN] Epoch=3/270, Step=28/45, loss=87.309227, lr=1.5e-05, time_each_step=0.66s, eta=2:18:32
    2021-08-15 12:28:41 [INFO]	[TRAIN] Epoch=3/270, Step=30/45, loss=88.609207, lr=1.5e-05, time_each_step=0.65s, eta=2:18:30
    2021-08-15 12:28:42 [INFO]	[TRAIN] Epoch=3/270, Step=32/45, loss=91.696678, lr=1.5e-05, time_each_step=0.65s, eta=2:18:29
    2021-08-15 12:28:43 [INFO]	[TRAIN] Epoch=3/270, Step=34/45, loss=91.133408, lr=1.5e-05, time_each_step=0.62s, eta=2:18:21
    2021-08-15 12:28:43 [INFO]	[TRAIN] Epoch=3/270, Step=36/45, loss=71.251236, lr=1.6e-05, time_each_step=0.57s, eta=2:18:12
    2021-08-15 12:28:44 [INFO]	[TRAIN] Epoch=3/270, Step=38/45, loss=84.505074, lr=1.6e-05, time_each_step=0.53s, eta=2:18:1
    2021-08-15 12:28:45 [INFO]	[TRAIN] Epoch=3/270, Step=40/45, loss=84.262131, lr=1.6e-05, time_each_step=0.5s, eta=2:17:55
    2021-08-15 12:28:45 [INFO]	[TRAIN] Epoch=3/270, Step=42/45, loss=79.80822, lr=1.6e-05, time_each_step=0.48s, eta=2:17:51
    2021-08-15 12:28:46 [INFO]	[TRAIN] Epoch=3/270, Step=44/45, loss=71.942886, lr=1.7e-05, time_each_step=0.45s, eta=2:17:44
    2021-08-15 12:28:46 [INFO]	[TRAIN] Epoch 3 finished, loss=94.41864, lr=1.4e-05 .
    2021-08-15 12:28:52 [INFO]	[TRAIN] Epoch=4/270, Step=1/45, loss=64.694473, lr=1.7e-05, time_each_step=0.69s, eta=2:38:55
    2021-08-15 12:28:54 [INFO]	[TRAIN] Epoch=4/270, Step=3/45, loss=67.127312, lr=1.7e-05, time_each_step=0.73s, eta=2:39:3
    2021-08-15 12:28:55 [INFO]	[TRAIN] Epoch=4/270, Step=5/45, loss=64.78643, lr=1.7e-05, time_each_step=0.74s, eta=2:39:4
    2021-08-15 12:28:57 [INFO]	[TRAIN] Epoch=4/270, Step=7/45, loss=61.134762, lr=1.8e-05, time_each_step=0.79s, eta=2:39:13
    2021-08-15 12:28:59 [INFO]	[TRAIN] Epoch=4/270, Step=9/45, loss=69.331848, lr=1.8e-05, time_each_step=0.81s, eta=2:39:17
    2021-08-15 12:29:00 [INFO]	[TRAIN] Epoch=4/270, Step=11/45, loss=51.24683, lr=1.8e-05, time_each_step=0.86s, eta=2:39:25
    2021-08-15 12:29:02 [INFO]	[TRAIN] Epoch=4/270, Step=13/45, loss=66.568138, lr=1.8e-05, time_each_step=0.92s, eta=2:39:36
    2021-08-15 12:29:04 [INFO]	[TRAIN] Epoch=4/270, Step=15/45, loss=62.625187, lr=1.9e-05, time_each_step=0.96s, eta=2:39:43
    2021-08-15 12:29:05 [INFO]	[TRAIN] Epoch=4/270, Step=17/45, loss=50.749866, lr=1.9e-05, time_each_step=0.96s, eta=2:39:42
    2021-08-15 12:29:07 [INFO]	[TRAIN] Epoch=4/270, Step=19/45, loss=49.12344, lr=1.9e-05, time_each_step=1.05s, eta=2:39:57
    2021-08-15 12:29:08 [INFO]	[TRAIN] Epoch=4/270, Step=21/45, loss=55.696785, lr=1.9e-05, time_each_step=0.79s, eta=2:39:3
    2021-08-15 12:29:09 [INFO]	[TRAIN] Epoch=4/270, Step=23/45, loss=52.183083, lr=2e-05, time_each_step=0.77s, eta=2:38:56
    2021-08-15 12:29:10 [INFO]	[TRAIN] Epoch=4/270, Step=25/45, loss=42.101025, lr=2e-05, time_each_step=0.74s, eta=2:38:48
    2021-08-15 12:29:11 [INFO]	[TRAIN] Epoch=4/270, Step=27/45, loss=44.570351, lr=2e-05, time_each_step=0.69s, eta=2:38:39
    2021-08-15 12:29:12 [INFO]	[TRAIN] Epoch=4/270, Step=29/45, loss=70.960785, lr=2e-05, time_each_step=0.67s, eta=2:38:33
    2021-08-15 12:29:14 [INFO]	[TRAIN] Epoch=4/270, Step=31/45, loss=47.156921, lr=2.1e-05, time_each_step=0.66s, eta=2:38:29
    2021-08-15 12:29:14 [INFO]	[TRAIN] Epoch=4/270, Step=33/45, loss=38.154652, lr=2.1e-05, time_each_step=0.62s, eta=2:38:19
    2021-08-15 12:29:15 [INFO]	[TRAIN] Epoch=4/270, Step=35/45, loss=43.659386, lr=2.1e-05, time_each_step=0.57s, eta=2:38:8
    2021-08-15 12:29:16 [INFO]	[TRAIN] Epoch=4/270, Step=37/45, loss=38.746643, lr=2.1e-05, time_each_step=0.56s, eta=2:38:6
    2021-08-15 12:29:17 [INFO]	[TRAIN] Epoch=4/270, Step=39/45, loss=36.655918, lr=2.2e-05, time_each_step=0.49s, eta=2:37:52
    2021-08-15 12:29:17 [INFO]	[TRAIN] Epoch=4/270, Step=41/45, loss=37.034958, lr=2.2e-05, time_each_step=0.45s, eta=2:37:44
    2021-08-15 12:29:18 [INFO]	[TRAIN] Epoch=4/270, Step=43/45, loss=41.282967, lr=2.2e-05, time_each_step=0.43s, eta=2:37:39
    2021-08-15 12:29:18 [INFO]	[TRAIN] Epoch=4/270, Step=45/45, loss=41.561214, lr=2.2e-05, time_each_step=0.41s, eta=2:37:34
    2021-08-15 12:29:18 [INFO]	[TRAIN] Epoch 4 finished, loss=52.886627, lr=2e-05 .
    2021-08-15 12:29:30 [INFO]	[TRAIN] Epoch=5/270, Step=2/45, loss=40.253391, lr=2.3e-05, time_each_step=0.92s, eta=2:25:40
    2021-08-15 12:29:31 [INFO]	[TRAIN] Epoch=5/270, Step=4/45, loss=37.185253, lr=2.3e-05, time_each_step=0.94s, eta=2:25:43
    2021-08-15 12:29:33 [INFO]	[TRAIN] Epoch=5/270, Step=6/45, loss=27.013401, lr=2.3e-05, time_each_step=0.95s, eta=2:25:43
    2021-08-15 12:29:34 [INFO]	[TRAIN] Epoch=5/270, Step=8/45, loss=29.492229, lr=2.3e-05, time_each_step=1.0s, eta=2:25:52
    2021-08-15 12:29:36 [INFO]	[TRAIN] Epoch=5/270, Step=10/45, loss=36.954834, lr=2.4e-05, time_each_step=1.06s, eta=2:26:2
    2021-08-15 12:29:38 [INFO]	[TRAIN] Epoch=5/270, Step=12/45, loss=29.195127, lr=2.4e-05, time_each_step=1.09s, eta=2:26:7
    2021-08-15 12:29:39 [INFO]	[TRAIN] Epoch=5/270, Step=14/45, loss=34.386833, lr=2.4e-05, time_each_step=1.15s, eta=2:26:17
    2021-08-15 12:29:41 [INFO]	[TRAIN] Epoch=5/270, Step=16/45, loss=34.800339, lr=2.4e-05, time_each_step=1.2s, eta=2:26:25
    2021-08-15 12:29:42 [INFO]	[TRAIN] Epoch=5/270, Step=18/45, loss=34.797352, lr=2.5e-05, time_each_step=1.22s, eta=2:26:28
    2021-08-15 12:29:43 [INFO]	[TRAIN] Epoch=5/270, Step=20/45, loss=30.646143, lr=2.5e-05, time_each_step=1.25s, eta=2:26:32
    2021-08-15 12:29:44 [INFO]	[TRAIN] Epoch=5/270, Step=22/45, loss=41.519936, lr=2.5e-05, time_each_step=0.73s, eta=2:24:42
    2021-08-15 12:29:46 [INFO]	[TRAIN] Epoch=5/270, Step=24/45, loss=33.283009, lr=2.5e-05, time_each_step=0.72s, eta=2:24:38
    2021-08-15 12:29:47 [INFO]	[TRAIN] Epoch=5/270, Step=26/45, loss=30.322058, lr=2.6e-05, time_each_step=0.7s, eta=2:24:34
    2021-08-15 12:29:48 [INFO]	[TRAIN] Epoch=5/270, Step=28/45, loss=20.283409, lr=2.6e-05, time_each_step=0.67s, eta=2:24:25
    2021-08-15 12:29:49 [INFO]	[TRAIN] Epoch=5/270, Step=30/45, loss=31.035223, lr=2.6e-05, time_each_step=0.63s, eta=2:24:17
    2021-08-15 12:29:50 [INFO]	[TRAIN] Epoch=5/270, Step=32/45, loss=27.036055, lr=2.6e-05, time_each_step=0.62s, eta=2:24:12
    2021-08-15 12:29:51 [INFO]	[TRAIN] Epoch=5/270, Step=34/45, loss=29.947371, lr=2.7e-05, time_each_step=0.56s, eta=2:24:1
    2021-08-15 12:29:51 [INFO]	[TRAIN] Epoch=5/270, Step=36/45, loss=26.050417, lr=2.7e-05, time_each_step=0.53s, eta=2:23:53
    2021-08-15 12:29:52 [INFO]	[TRAIN] Epoch=5/270, Step=38/45, loss=29.338474, lr=2.7e-05, time_each_step=0.49s, eta=2:23:45
    2021-08-15 12:29:53 [INFO]	[TRAIN] Epoch=5/270, Step=40/45, loss=37.069233, lr=2.7e-05, time_each_step=0.47s, eta=2:23:39
    2021-08-15 12:29:53 [INFO]	[TRAIN] Epoch=5/270, Step=42/45, loss=27.694641, lr=2.8e-05, time_each_step=0.44s, eta=2:23:33
    2021-08-15 12:29:54 [INFO]	[TRAIN] Epoch=5/270, Step=44/45, loss=26.741968, lr=2.8e-05, time_each_step=0.41s, eta=2:23:28
    2021-08-15 12:29:54 [INFO]	[TRAIN] Epoch 5 finished, loss=31.091362, lr=2.5e-05 .
    2021-08-15 12:29:59 [INFO]	[TRAIN] Epoch=6/270, Step=1/45, loss=27.053427, lr=2.8e-05, time_each_step=0.63s, eta=2:39:25
    2021-08-15 12:30:02 [INFO]	[TRAIN] Epoch=6/270, Step=3/45, loss=26.182304, lr=2.8e-05, time_each_step=0.69s, eta=2:39:37
    2021-08-15 12:30:04 [INFO]	[TRAIN] Epoch=6/270, Step=5/45, loss=20.560179, lr=2.9e-05, time_each_step=0.74s, eta=2:39:46
    2021-08-15 12:30:06 [INFO]	[TRAIN] Epoch=6/270, Step=7/45, loss=24.657106, lr=2.9e-05, time_each_step=0.78s, eta=2:39:55
    2021-08-15 12:30:07 [INFO]	[TRAIN] Epoch=6/270, Step=9/45, loss=22.618252, lr=2.9e-05, time_each_step=0.83s, eta=2:40:3
    2021-08-15 12:30:09 [INFO]	[TRAIN] Epoch=6/270, Step=11/45, loss=26.038841, lr=2.9e-05, time_each_step=0.87s, eta=2:40:11
    2021-08-15 12:30:11 [INFO]	[TRAIN] Epoch=6/270, Step=13/45, loss=22.578894, lr=3e-05, time_each_step=0.94s, eta=2:40:23
    2021-08-15 12:30:12 [INFO]	[TRAIN] Epoch=6/270, Step=15/45, loss=22.820374, lr=3e-05, time_each_step=0.97s, eta=2:40:28
    2021-08-15 12:30:13 [INFO]	[TRAIN] Epoch=6/270, Step=17/45, loss=26.124557, lr=3e-05, time_each_step=1.0s, eta=2:40:33
    2021-08-15 12:30:14 [INFO]	[TRAIN] Epoch=6/270, Step=19/45, loss=24.3535, lr=3e-05, time_each_step=1.02s, eta=2:40:35
    2021-08-15 12:30:16 [INFO]	[TRAIN] Epoch=6/270, Step=21/45, loss=24.930292, lr=3.1e-05, time_each_step=0.82s, eta=2:39:50
    2021-08-15 12:30:17 [INFO]	[TRAIN] Epoch=6/270, Step=23/45, loss=22.656851, lr=3.1e-05, time_each_step=0.75s, eta=2:39:36
    2021-08-15 12:30:18 [INFO]	[TRAIN] Epoch=6/270, Step=25/45, loss=30.019999, lr=3.1e-05, time_each_step=0.72s, eta=2:39:27
    2021-08-15 12:30:19 [INFO]	[TRAIN] Epoch=6/270, Step=27/45, loss=19.751732, lr=3.1e-05, time_each_step=0.68s, eta=2:39:18
    2021-08-15 12:30:20 [INFO]	[TRAIN] Epoch=6/270, Step=29/45, loss=19.488848, lr=3.2e-05, time_each_step=0.65s, eta=2:39:12
    2021-08-15 12:30:21 [INFO]	[TRAIN] Epoch=6/270, Step=31/45, loss=22.127382, lr=3.2e-05, time_each_step=0.62s, eta=2:39:3
    2021-08-15 12:30:22 [INFO]	[TRAIN] Epoch=6/270, Step=33/45, loss=24.811037, lr=3.2e-05, time_each_step=0.57s, eta=2:38:53
    2021-08-15 12:30:22 [INFO]	[TRAIN] Epoch=6/270, Step=35/45, loss=22.41827, lr=3.2e-05, time_each_step=0.52s, eta=2:38:42
    2021-08-15 12:30:23 [INFO]	[TRAIN] Epoch=6/270, Step=37/45, loss=22.067917, lr=3.3e-05, time_each_step=0.49s, eta=2:38:35
    2021-08-15 12:30:24 [INFO]	[TRAIN] Epoch=6/270, Step=39/45, loss=20.163944, lr=3.3e-05, time_each_step=0.46s, eta=2:38:29
    2021-08-15 12:30:24 [INFO]	[TRAIN] Epoch=6/270, Step=41/45, loss=26.717596, lr=3.3e-05, time_each_step=0.42s, eta=2:38:20
    2021-08-15 12:30:25 [INFO]	[TRAIN] Epoch=6/270, Step=43/45, loss=22.061253, lr=3.3e-05, time_each_step=0.4s, eta=2:38:16
    2021-08-15 12:30:25 [INFO]	[TRAIN] Epoch=6/270, Step=45/45, loss=22.555542, lr=3.4e-05, time_each_step=0.37s, eta=2:38:9
    2021-08-15 12:30:25 [INFO]	[TRAIN] Epoch 6 finished, loss=23.976711, lr=3.1e-05 .
    2021-08-15 12:30:39 [INFO]	[TRAIN] Epoch=7/270, Step=2/45, loss=29.723394, lr=3.4e-05, time_each_step=0.99s, eta=2:20:52
    2021-08-15 12:30:41 [INFO]	[TRAIN] Epoch=7/270, Step=4/45, loss=25.684227, lr=3.4e-05, time_each_step=1.03s, eta=2:20:59
    2021-08-15 12:30:44 [INFO]	[TRAIN] Epoch=7/270, Step=6/45, loss=23.30596, lr=3.4e-05, time_each_step=1.12s, eta=2:21:16
    2021-08-15 12:30:45 [INFO]	[TRAIN] Epoch=7/270, Step=8/45, loss=31.301954, lr=3.5e-05, time_each_step=1.17s, eta=2:21:25
    2021-08-15 12:30:46 [INFO]	[TRAIN] Epoch=7/270, Step=10/45, loss=26.895128, lr=3.5e-05, time_each_step=1.2s, eta=2:21:30
    2021-08-15 12:30:48 [INFO]	[TRAIN] Epoch=7/270, Step=12/45, loss=27.554264, lr=3.5e-05, time_each_step=1.24s, eta=2:21:35
    2021-08-15 12:30:49 [INFO]	[TRAIN] Epoch=7/270, Step=14/45, loss=20.729404, lr=3.5e-05, time_each_step=1.27s, eta=2:21:39
    2021-08-15 12:30:50 [INFO]	[TRAIN] Epoch=7/270, Step=16/45, loss=18.65192, lr=3.6e-05, time_each_step=1.31s, eta=2:21:45
    2021-08-15 12:30:51 [INFO]	[TRAIN] Epoch=7/270, Step=18/45, loss=22.953543, lr=3.6e-05, time_each_step=1.34s, eta=2:21:49
    2021-08-15 12:30:53 [INFO]	[TRAIN] Epoch=7/270, Step=20/45, loss=24.348274, lr=3.6e-05, time_each_step=1.36s, eta=2:21:52
    2021-08-15 12:30:54 [INFO]	[TRAIN] Epoch=7/270, Step=22/45, loss=17.8764, lr=3.6e-05, time_each_step=0.75s, eta=2:19:43
    2021-08-15 12:30:55 [INFO]	[TRAIN] Epoch=7/270, Step=24/45, loss=21.001282, lr=3.7e-05, time_each_step=0.71s, eta=2:19:34
    2021-08-15 12:30:57 [INFO]	[TRAIN] Epoch=7/270, Step=26/45, loss=20.693155, lr=3.7e-05, time_each_step=0.65s, eta=2:19:21
    2021-08-15 12:30:58 [INFO]	[TRAIN] Epoch=7/270, Step=28/45, loss=18.129642, lr=3.7e-05, time_each_step=0.61s, eta=2:19:11
    2021-08-15 12:30:59 [INFO]	[TRAIN] Epoch=7/270, Step=30/45, loss=18.220921, lr=3.7e-05, time_each_step=0.61s, eta=2:19:10
    2021-08-15 12:31:00 [INFO]	[TRAIN] Epoch=7/270, Step=32/45, loss=29.059727, lr=3.8e-05, time_each_step=0.59s, eta=2:19:4
    2021-08-15 12:31:00 [INFO]	[TRAIN] Epoch=7/270, Step=34/45, loss=23.524317, lr=3.8e-05, time_each_step=0.57s, eta=2:18:59
    2021-08-15 12:31:01 [INFO]	[TRAIN] Epoch=7/270, Step=36/45, loss=13.752151, lr=3.8e-05, time_each_step=0.52s, eta=2:18:49
    2021-08-15 12:31:01 [INFO]	[TRAIN] Epoch=7/270, Step=38/45, loss=20.177597, lr=3.8e-05, time_each_step=0.49s, eta=2:18:41
    2021-08-15 12:31:02 [INFO]	[TRAIN] Epoch=7/270, Step=40/45, loss=18.332333, lr=3.9e-05, time_each_step=0.45s, eta=2:18:34
    2021-08-15 12:31:02 [INFO]	[TRAIN] Epoch=7/270, Step=42/45, loss=20.416582, lr=3.9e-05, time_each_step=0.42s, eta=2:18:28
    2021-08-15 12:31:03 [INFO]	[TRAIN] Epoch=7/270, Step=44/45, loss=19.661064, lr=3.9e-05, time_each_step=0.38s, eta=2:18:19
    2021-08-15 12:31:03 [INFO]	[TRAIN] Epoch 7 finished, loss=22.203131, lr=3.7e-05 .
    2021-08-15 12:31:09 [INFO]	[TRAIN] Epoch=8/270, Step=1/45, loss=18.577534, lr=3.9e-05, time_each_step=0.6s, eta=2:46:52
    2021-08-15 12:31:10 [INFO]	[TRAIN] Epoch=8/270, Step=3/45, loss=19.527195, lr=4e-05, time_each_step=0.64s, eta=2:46:59
    2021-08-15 12:31:12 [INFO]	[TRAIN] Epoch=8/270, Step=5/45, loss=17.786592, lr=4e-05, time_each_step=0.69s, eta=2:47:8
    2021-08-15 12:31:14 [INFO]	[TRAIN] Epoch=8/270, Step=7/45, loss=17.858917, lr=4e-05, time_each_step=0.72s, eta=2:47:13
    2021-08-15 12:31:15 [INFO]	[TRAIN] Epoch=8/270, Step=9/45, loss=18.204744, lr=4e-05, time_each_step=0.76s, eta=2:47:20
    2021-08-15 12:31:17 [INFO]	[TRAIN] Epoch=8/270, Step=11/45, loss=19.980085, lr=4.1e-05, time_each_step=0.83s, eta=2:47:34
    2021-08-15 12:31:19 [INFO]	[TRAIN] Epoch=8/270, Step=13/45, loss=18.43198, lr=4.1e-05, time_each_step=0.88s, eta=2:47:43
    2021-08-15 12:31:20 [INFO]	[TRAIN] Epoch=8/270, Step=15/45, loss=19.758417, lr=4.1e-05, time_each_step=0.93s, eta=2:47:53
    2021-08-15 12:31:22 [INFO]	[TRAIN] Epoch=8/270, Step=17/45, loss=20.842785, lr=4.1e-05, time_each_step=0.97s, eta=2:47:59
    2021-08-15 12:31:23 [INFO]	[TRAIN] Epoch=8/270, Step=19/45, loss=17.223358, lr=4.2e-05, time_each_step=1.0s, eta=2:48:4
    2021-08-15 12:31:24 [INFO]	[TRAIN] Epoch=8/270, Step=21/45, loss=21.312342, lr=4.2e-05, time_each_step=0.76s, eta=2:47:12
    2021-08-15 12:31:25 [INFO]	[TRAIN] Epoch=8/270, Step=23/45, loss=20.09576, lr=4.2e-05, time_each_step=0.74s, eta=2:47:6
    2021-08-15 12:31:26 [INFO]	[TRAIN] Epoch=8/270, Step=25/45, loss=16.1847, lr=4.2e-05, time_each_step=0.7s, eta=2:46:57
    2021-08-15 12:31:27 [INFO]	[TRAIN] Epoch=8/270, Step=27/45, loss=23.816671, lr=4.3e-05, time_each_step=0.68s, eta=2:46:52
    2021-08-15 12:31:29 [INFO]	[TRAIN] Epoch=8/270, Step=29/45, loss=24.95756, lr=4.3e-05, time_each_step=0.67s, eta=2:46:47
    2021-08-15 12:31:30 [INFO]	[TRAIN] Epoch=8/270, Step=31/45, loss=16.806843, lr=4.3e-05, time_each_step=0.62s, eta=2:46:38
    2021-08-15 12:31:31 [INFO]	[TRAIN] Epoch=8/270, Step=33/45, loss=19.519537, lr=4.3e-05, time_each_step=0.6s, eta=2:46:32
    2021-08-15 12:31:31 [INFO]	[TRAIN] Epoch=8/270, Step=35/45, loss=18.609503, lr=4.4e-05, time_each_step=0.55s, eta=2:46:22
    2021-08-15 12:31:32 [INFO]	[TRAIN] Epoch=8/270, Step=37/45, loss=22.358986, lr=4.4e-05, time_each_step=0.51s, eta=2:46:12
    2021-08-15 12:31:32 [INFO]	[TRAIN] Epoch=8/270, Step=39/45, loss=16.718679, lr=4.4e-05, time_each_step=0.48s, eta=2:46:5
    2021-08-15 12:31:33 [INFO]	[TRAIN] Epoch=8/270, Step=41/45, loss=23.985878, lr=4.4e-05, time_each_step=0.45s, eta=2:46:0
    2021-08-15 12:31:34 [INFO]	[TRAIN] Epoch=8/270, Step=43/45, loss=13.520609, lr=4.5e-05, time_each_step=0.42s, eta=2:45:53
    2021-08-15 12:31:34 [INFO]	[TRAIN] Epoch=8/270, Step=45/45, loss=19.137411, lr=4.5e-05, time_each_step=0.39s, eta=2:45:46
    2021-08-15 12:31:34 [INFO]	[TRAIN] Epoch 8 finished, loss=19.654068, lr=4.2e-05 .
    2021-08-15 12:31:46 [INFO]	[TRAIN] Epoch=9/270, Step=2/45, loss=16.946136, lr=4.5e-05, time_each_step=0.94s, eta=2:19:21
    2021-08-15 12:31:48 [INFO]	[TRAIN] Epoch=9/270, Step=4/45, loss=17.133873, lr=4.5e-05, time_each_step=0.96s, eta=2:19:24
    2021-08-15 12:31:50 [INFO]	[TRAIN] Epoch=9/270, Step=6/45, loss=14.40705, lr=4.6e-05, time_each_step=1.01s, eta=2:19:32
    2021-08-15 12:31:51 [INFO]	[TRAIN] Epoch=9/270, Step=8/45, loss=17.793844, lr=4.6e-05, time_each_step=1.03s, eta=2:19:35
    2021-08-15 12:31:53 [INFO]	[TRAIN] Epoch=9/270, Step=10/45, loss=20.019419, lr=4.6e-05, time_each_step=1.09s, eta=2:19:47
    2021-08-15 12:31:55 [INFO]	[TRAIN] Epoch=9/270, Step=12/45, loss=17.830547, lr=4.6e-05, time_each_step=1.16s, eta=2:20:0
    2021-08-15 12:31:57 [INFO]	[TRAIN] Epoch=9/270, Step=14/45, loss=20.119755, lr=4.7e-05, time_each_step=1.22s, eta=2:20:10
    2021-08-15 12:31:58 [INFO]	[TRAIN] Epoch=9/270, Step=16/45, loss=38.727352, lr=4.7e-05, time_each_step=1.24s, eta=2:20:12
    2021-08-15 12:31:59 [INFO]	[TRAIN] Epoch=9/270, Step=18/45, loss=12.921808, lr=4.7e-05, time_each_step=1.28s, eta=2:20:16
    2021-08-15 12:32:01 [INFO]	[TRAIN] Epoch=9/270, Step=20/45, loss=17.039005, lr=4.7e-05, time_each_step=1.32s, eta=2:20:23
    2021-08-15 12:32:02 [INFO]	[TRAIN] Epoch=9/270, Step=22/45, loss=21.762363, lr=4.8e-05, time_each_step=0.77s, eta=2:18:27
    2021-08-15 12:32:03 [INFO]	[TRAIN] Epoch=9/270, Step=24/45, loss=15.92203, lr=4.8e-05, time_each_step=0.74s, eta=2:18:19
    2021-08-15 12:32:04 [INFO]	[TRAIN] Epoch=9/270, Step=26/45, loss=17.932114, lr=4.8e-05, time_each_step=0.7s, eta=2:18:11
    2021-08-15 12:32:05 [INFO]	[TRAIN] Epoch=9/270, Step=28/45, loss=15.860455, lr=4.8e-05, time_each_step=0.68s, eta=2:18:5
    2021-08-15 12:32:06 [INFO]	[TRAIN] Epoch=9/270, Step=30/45, loss=14.041147, lr=4.9e-05, time_each_step=0.64s, eta=2:17:57
    2021-08-15 12:32:07 [INFO]	[TRAIN] Epoch=9/270, Step=32/45, loss=15.192682, lr=4.9e-05, time_each_step=0.61s, eta=2:17:49
    2021-08-15 12:32:08 [INFO]	[TRAIN] Epoch=9/270, Step=34/45, loss=15.601509, lr=4.9e-05, time_each_step=0.56s, eta=2:17:38
    2021-08-15 12:32:09 [INFO]	[TRAIN] Epoch=9/270, Step=36/45, loss=20.06172, lr=4.9e-05, time_each_step=0.54s, eta=2:17:32
    2021-08-15 12:32:09 [INFO]	[TRAIN] Epoch=9/270, Step=38/45, loss=16.000088, lr=5e-05, time_each_step=0.5s, eta=2:17:25
    2021-08-15 12:32:10 [INFO]	[TRAIN] Epoch=9/270, Step=40/45, loss=12.632515, lr=5e-05, time_each_step=0.46s, eta=2:17:16
    2021-08-15 12:32:10 [INFO]	[TRAIN] Epoch=9/270, Step=42/45, loss=18.455378, lr=5e-05, time_each_step=0.44s, eta=2:17:11
    2021-08-15 12:32:11 [INFO]	[TRAIN] Epoch=9/270, Step=44/45, loss=18.504293, lr=5e-05, time_each_step=0.42s, eta=2:17:7
    2021-08-15 12:32:11 [INFO]	[TRAIN] Epoch 9 finished, loss=18.392342, lr=4.8e-05 .
    2021-08-15 12:32:17 [INFO]	[TRAIN] Epoch=10/270, Step=1/45, loss=16.866802, lr=5.1e-05, time_each_step=0.64s, eta=2:43:25
    2021-08-15 12:32:18 [INFO]	[TRAIN] Epoch=10/270, Step=3/45, loss=18.106552, lr=5.1e-05, time_each_step=0.67s, eta=2:43:31
    2021-08-15 12:32:20 [INFO]	[TRAIN] Epoch=10/270, Step=5/45, loss=15.127789, lr=5.1e-05, time_each_step=0.69s, eta=2:43:34
    2021-08-15 12:32:22 [INFO]	[TRAIN] Epoch=10/270, Step=7/45, loss=24.39673, lr=5.1e-05, time_each_step=0.75s, eta=2:43:46
    2021-08-15 12:32:24 [INFO]	[TRAIN] Epoch=10/270, Step=9/45, loss=19.94916, lr=5.2e-05, time_each_step=0.79s, eta=2:43:53
    2021-08-15 12:32:25 [INFO]	[TRAIN] Epoch=10/270, Step=11/45, loss=19.027699, lr=5.2e-05, time_each_step=0.81s, eta=2:43:56
    2021-08-15 12:32:27 [INFO]	[TRAIN] Epoch=10/270, Step=13/45, loss=21.380703, lr=5.2e-05, time_each_step=0.9s, eta=2:44:12
    2021-08-15 12:32:28 [INFO]	[TRAIN] Epoch=10/270, Step=15/45, loss=20.626694, lr=5.2e-05, time_each_step=0.92s, eta=2:44:16
    2021-08-15 12:32:29 [INFO]	[TRAIN] Epoch=10/270, Step=17/45, loss=14.897019, lr=5.3e-05, time_each_step=0.96s, eta=2:44:21
    2021-08-15 12:32:30 [INFO]	[TRAIN] Epoch=10/270, Step=19/45, loss=17.829624, lr=5.3e-05, time_each_step=0.97s, eta=2:44:23
    2021-08-15 12:32:32 [INFO]	[TRAIN] Epoch=10/270, Step=21/45, loss=19.467945, lr=5.3e-05, time_each_step=0.76s, eta=2:43:36
    2021-08-15 12:32:33 [INFO]	[TRAIN] Epoch=10/270, Step=23/45, loss=19.426556, lr=5.3e-05, time_each_step=0.73s, eta=2:43:28
    2021-08-15 12:32:34 [INFO]	[TRAIN] Epoch=10/270, Step=25/45, loss=16.706476, lr=5.4e-05, time_each_step=0.71s, eta=2:43:24
    2021-08-15 12:32:35 [INFO]	[TRAIN] Epoch=10/270, Step=27/45, loss=14.670384, lr=5.4e-05, time_each_step=0.63s, eta=2:43:6
    2021-08-15 12:32:36 [INFO]	[TRAIN] Epoch=10/270, Step=29/45, loss=14.419385, lr=5.4e-05, time_each_step=0.6s, eta=2:42:59
    2021-08-15 12:32:37 [INFO]	[TRAIN] Epoch=10/270, Step=31/45, loss=16.019163, lr=5.4e-05, time_each_step=0.6s, eta=2:42:59
    2021-08-15 12:32:38 [INFO]	[TRAIN] Epoch=10/270, Step=33/45, loss=15.596306, lr=5.5e-05, time_each_step=0.54s, eta=2:42:44
    2021-08-15 12:32:39 [INFO]	[TRAIN] Epoch=10/270, Step=35/45, loss=16.304115, lr=5.5e-05, time_each_step=0.52s, eta=2:42:40
    2021-08-15 12:32:39 [INFO]	[TRAIN] Epoch=10/270, Step=37/45, loss=14.648934, lr=5.5e-05, time_each_step=0.48s, eta=2:42:31
    2021-08-15 12:32:40 [INFO]	[TRAIN] Epoch=10/270, Step=39/45, loss=19.425629, lr=5.5e-05, time_each_step=0.46s, eta=2:42:27
    2021-08-15 12:32:40 [INFO]	[TRAIN] Epoch=10/270, Step=41/45, loss=14.469853, lr=5.6e-05, time_each_step=0.43s, eta=2:42:20
    2021-08-15 12:32:41 [INFO]	[TRAIN] Epoch=10/270, Step=43/45, loss=17.753262, lr=5.6e-05, time_each_step=0.4s, eta=2:42:14
    2021-08-15 12:32:42 [INFO]	[TRAIN] Epoch=10/270, Step=45/45, loss=15.28943, lr=5.6e-05, time_each_step=0.37s, eta=2:42:8
    2021-08-15 12:32:42 [INFO]	[TRAIN] Epoch 10 finished, loss=18.144218, lr=5.3e-05 .
    2021-08-15 12:32:50 [INFO]	[TRAIN] Epoch=11/270, Step=2/45, loss=22.53957, lr=5.6e-05, time_each_step=0.76s, eta=2:13:33
    2021-08-15 12:32:52 [INFO]	[TRAIN] Epoch=11/270, Step=4/45, loss=13.631577, lr=5.7e-05, time_each_step=0.81s, eta=2:13:43
    2021-08-15 12:32:54 [INFO]	[TRAIN] Epoch=11/270, Step=6/45, loss=18.135069, lr=5.7e-05, time_each_step=0.84s, eta=2:13:48
    2021-08-15 12:32:55 [INFO]	[TRAIN] Epoch=11/270, Step=8/45, loss=18.433428, lr=5.7e-05, time_each_step=0.87s, eta=2:13:51
    2021-08-15 12:32:56 [INFO]	[TRAIN] Epoch=11/270, Step=10/45, loss=15.378572, lr=5.7e-05, time_each_step=0.89s, eta=2:13:55
    2021-08-15 12:32:58 [INFO]	[TRAIN] Epoch=11/270, Step=12/45, loss=20.117273, lr=5.8e-05, time_each_step=0.96s, eta=2:14:8
    2021-08-15 12:33:00 [INFO]	[TRAIN] Epoch=11/270, Step=14/45, loss=18.083427, lr=5.8e-05, time_each_step=1.0s, eta=2:14:14
    2021-08-15 12:33:01 [INFO]	[TRAIN] Epoch=11/270, Step=16/45, loss=21.664377, lr=5.8e-05, time_each_step=1.03s, eta=2:14:19
    2021-08-15 12:33:02 [INFO]	[TRAIN] Epoch=11/270, Step=18/45, loss=16.658457, lr=5.8e-05, time_each_step=1.05s, eta=2:14:22
    2021-08-15 12:33:03 [INFO]	[TRAIN] Epoch=11/270, Step=20/45, loss=13.479628, lr=5.9e-05, time_each_step=1.08s, eta=2:14:25
    2021-08-15 12:33:04 [INFO]	[TRAIN] Epoch=11/270, Step=22/45, loss=37.945747, lr=5.9e-05, time_each_step=0.7s, eta=2:13:5
    2021-08-15 12:33:05 [INFO]	[TRAIN] Epoch=11/270, Step=24/45, loss=20.232347, lr=5.9e-05, time_each_step=0.65s, eta=2:12:53
    2021-08-15 12:33:06 [INFO]	[TRAIN] Epoch=11/270, Step=26/45, loss=18.184685, lr=5.9e-05, time_each_step=0.61s, eta=2:12:45
    2021-08-15 12:33:07 [INFO]	[TRAIN] Epoch=11/270, Step=28/45, loss=22.229307, lr=6e-05, time_each_step=0.61s, eta=2:12:42
    2021-08-15 12:33:08 [INFO]	[TRAIN] Epoch=11/270, Step=30/45, loss=23.563232, lr=6e-05, time_each_step=0.59s, eta=2:12:37
    2021-08-15 12:33:09 [INFO]	[TRAIN] Epoch=11/270, Step=32/45, loss=19.865477, lr=6e-05, time_each_step=0.55s, eta=2:12:28
    2021-08-15 12:33:10 [INFO]	[TRAIN] Epoch=11/270, Step=34/45, loss=13.457856, lr=6e-05, time_each_step=0.52s, eta=2:12:21
    2021-08-15 12:33:11 [INFO]	[TRAIN] Epoch=11/270, Step=36/45, loss=14.703779, lr=6.1e-05, time_each_step=0.48s, eta=2:12:14
    2021-08-15 12:33:11 [INFO]	[TRAIN] Epoch=11/270, Step=38/45, loss=24.811413, lr=6.1e-05, time_each_step=0.44s, eta=2:12:6
    2021-08-15 12:33:12 [INFO]	[TRAIN] Epoch=11/270, Step=40/45, loss=13.803166, lr=6.1e-05, time_each_step=0.42s, eta=2:11:59
    2021-08-15 12:33:12 [INFO]	[TRAIN] Epoch=11/270, Step=42/45, loss=16.048504, lr=6.1e-05, time_each_step=0.38s, eta=2:11:52
    2021-08-15 12:33:12 [INFO]	[TRAIN] Epoch=11/270, Step=44/45, loss=21.488697, lr=6.2e-05, time_each_step=0.36s, eta=2:11:48
    2021-08-15 12:33:13 [INFO]	[TRAIN] Epoch 11 finished, loss=18.817238, lr=5.9e-05 .
    2021-08-15 12:33:18 [INFO]	[TRAIN] Epoch=12/270, Step=1/45, loss=15.121563, lr=6.2e-05, time_each_step=0.58s, eta=2:16:12
    2021-08-15 12:33:20 [INFO]	[TRAIN] Epoch=12/270, Step=3/45, loss=15.258106, lr=6.2e-05, time_each_step=0.62s, eta=2:16:19
    2021-08-15 12:33:22 [INFO]	[TRAIN] Epoch=12/270, Step=5/45, loss=15.100964, lr=6.2e-05, time_each_step=0.71s, eta=2:16:36
    2021-08-15 12:33:24 [INFO]	[TRAIN] Epoch=12/270, Step=7/45, loss=13.194704, lr=6.3e-05, time_each_step=0.74s, eta=2:16:44
    2021-08-15 12:33:26 [INFO]	[TRAIN] Epoch=12/270, Step=9/45, loss=34.471596, lr=6.3e-05, time_each_step=0.79s, eta=2:16:53
    2021-08-15 12:33:27 [INFO]	[TRAIN] Epoch=12/270, Step=11/45, loss=11.900182, lr=6.3e-05, time_each_step=0.84s, eta=2:17:2
    2021-08-15 12:33:29 [INFO]	[TRAIN] Epoch=12/270, Step=13/45, loss=15.559011, lr=6.3e-05, time_each_step=0.89s, eta=2:17:10
    2021-08-15 12:33:30 [INFO]	[TRAIN] Epoch=12/270, Step=15/45, loss=16.185083, lr=6.4e-05, time_each_step=0.92s, eta=2:17:14
    2021-08-15 12:33:31 [INFO]	[TRAIN] Epoch=12/270, Step=17/45, loss=14.086213, lr=6.4e-05, time_each_step=0.95s, eta=2:17:20
    2021-08-15 12:33:32 [INFO]	[TRAIN] Epoch=12/270, Step=19/45, loss=17.290615, lr=6.4e-05, time_each_step=0.99s, eta=2:17:26
    2021-08-15 12:33:33 [INFO]	[TRAIN] Epoch=12/270, Step=21/45, loss=14.600842, lr=6.4e-05, time_each_step=0.79s, eta=2:16:42
    2021-08-15 12:33:34 [INFO]	[TRAIN] Epoch=12/270, Step=23/45, loss=12.188151, lr=6.5e-05, time_each_step=0.74s, eta=2:16:30
    2021-08-15 12:33:36 [INFO]	[TRAIN] Epoch=12/270, Step=25/45, loss=14.638754, lr=6.5e-05, time_each_step=0.67s, eta=2:16:15
    2021-08-15 12:33:37 [INFO]	[TRAIN] Epoch=12/270, Step=27/45, loss=13.948963, lr=6.5e-05, time_each_step=0.63s, eta=2:16:5
    2021-08-15 12:33:38 [INFO]	[TRAIN] Epoch=12/270, Step=29/45, loss=16.454477, lr=6.5e-05, time_each_step=0.6s, eta=2:15:59
    2021-08-15 12:33:39 [INFO]	[TRAIN] Epoch=12/270, Step=31/45, loss=17.518381, lr=6.6e-05, time_each_step=0.56s, eta=2:15:50
    2021-08-15 12:33:40 [INFO]	[TRAIN] Epoch=12/270, Step=33/45, loss=14.981278, lr=6.6e-05, time_each_step=0.54s, eta=2:15:44
    2021-08-15 12:33:40 [INFO]	[TRAIN] Epoch=12/270, Step=35/45, loss=18.058418, lr=6.6e-05, time_each_step=0.51s, eta=2:15:38
    2021-08-15 12:33:41 [INFO]	[TRAIN] Epoch=12/270, Step=37/45, loss=15.800297, lr=6.6e-05, time_each_step=0.49s, eta=2:15:34
    2021-08-15 12:33:41 [INFO]	[TRAIN] Epoch=12/270, Step=39/45, loss=16.525574, lr=6.7e-05, time_each_step=0.46s, eta=2:15:25
    2021-08-15 12:33:42 [INFO]	[TRAIN] Epoch=12/270, Step=41/45, loss=17.12035, lr=6.7e-05, time_each_step=0.42s, eta=2:15:18
    2021-08-15 12:33:42 [INFO]	[TRAIN] Epoch=12/270, Step=43/45, loss=15.241709, lr=6.7e-05, time_each_step=0.4s, eta=2:15:13
    2021-08-15 12:33:43 [INFO]	[TRAIN] Epoch=12/270, Step=45/45, loss=13.345188, lr=6.7e-05, time_each_step=0.36s, eta=2:15:5
    2021-08-15 12:33:43 [INFO]	[TRAIN] Epoch 12 finished, loss=16.149206, lr=6.5e-05 .
    2021-08-15 12:33:54 [INFO]	[TRAIN] Epoch=13/270, Step=2/45, loss=13.949957, lr=6.8e-05, time_each_step=0.87s, eta=2:11:50
    2021-08-15 12:33:56 [INFO]	[TRAIN] Epoch=13/270, Step=4/45, loss=22.376499, lr=6.8e-05, time_each_step=0.91s, eta=2:11:56
    2021-08-15 12:33:58 [INFO]	[TRAIN] Epoch=13/270, Step=6/45, loss=12.032713, lr=6.8e-05, time_each_step=0.94s, eta=2:12:2
    2021-08-15 12:33:59 [INFO]	[TRAIN] Epoch=13/270, Step=8/45, loss=17.063675, lr=6.8e-05, time_each_step=0.97s, eta=2:12:6
    2021-08-15 12:34:00 [INFO]	[TRAIN] Epoch=13/270, Step=10/45, loss=13.850479, lr=6.9e-05, time_each_step=1.02s, eta=2:12:14
    2021-08-15 12:34:02 [INFO]	[TRAIN] Epoch=13/270, Step=12/45, loss=15.408713, lr=6.9e-05, time_each_step=1.06s, eta=2:12:22
    2021-08-15 12:34:03 [INFO]	[TRAIN] Epoch=13/270, Step=14/45, loss=15.950863, lr=6.9e-05, time_each_step=1.1s, eta=2:12:28
    2021-08-15 12:34:04 [INFO]	[TRAIN] Epoch=13/270, Step=16/45, loss=12.376808, lr=6.9e-05, time_each_step=1.13s, eta=2:12:31
    2021-08-15 12:34:06 [INFO]	[TRAIN] Epoch=13/270, Step=18/45, loss=12.863141, lr=7e-05, time_each_step=1.16s, eta=2:12:37
    2021-08-15 12:34:07 [INFO]	[TRAIN] Epoch=13/270, Step=20/45, loss=19.459969, lr=7e-05, time_each_step=1.19s, eta=2:12:39
    2021-08-15 12:34:08 [INFO]	[TRAIN] Epoch=13/270, Step=22/45, loss=19.74036, lr=7e-05, time_each_step=0.68s, eta=2:10:54
    2021-08-15 12:34:09 [INFO]	[TRAIN] Epoch=13/270, Step=24/45, loss=13.25842, lr=7e-05, time_each_step=0.65s, eta=2:10:45
    2021-08-15 12:34:10 [INFO]	[TRAIN] Epoch=13/270, Step=26/45, loss=21.126675, lr=7.1e-05, time_each_step=0.62s, eta=2:10:38
    2021-08-15 12:34:11 [INFO]	[TRAIN] Epoch=13/270, Step=28/45, loss=24.064419, lr=7.1e-05, time_each_step=0.61s, eta=2:10:34
    2021-08-15 12:34:12 [INFO]	[TRAIN] Epoch=13/270, Step=30/45, loss=14.321312, lr=7.1e-05, time_each_step=0.58s, eta=2:10:27
    2021-08-15 12:34:13 [INFO]	[TRAIN] Epoch=13/270, Step=32/45, loss=18.691324, lr=7.1e-05, time_each_step=0.54s, eta=2:10:19
    2021-08-15 12:34:14 [INFO]	[TRAIN] Epoch=13/270, Step=34/45, loss=14.555198, lr=7.2e-05, time_each_step=0.51s, eta=2:10:12
    2021-08-15 12:34:14 [INFO]	[TRAIN] Epoch=13/270, Step=36/45, loss=14.429372, lr=7.2e-05, time_each_step=0.48s, eta=2:10:5
    2021-08-15 12:34:15 [INFO]	[TRAIN] Epoch=13/270, Step=38/45, loss=17.64525, lr=7.2e-05, time_each_step=0.45s, eta=2:9:58
    2021-08-15 12:34:15 [INFO]	[TRAIN] Epoch=13/270, Step=40/45, loss=14.654922, lr=7.2e-05, time_each_step=0.44s, eta=2:9:56
    2021-08-15 12:34:16 [INFO]	[TRAIN] Epoch=13/270, Step=42/45, loss=18.097883, lr=7.3e-05, time_each_step=0.41s, eta=2:9:48
    2021-08-15 12:34:16 [INFO]	[TRAIN] Epoch=13/270, Step=44/45, loss=17.600882, lr=7.3e-05, time_each_step=0.37s, eta=2:9:40
    2021-08-15 12:34:16 [INFO]	[TRAIN] Epoch 13 finished, loss=16.02347, lr=7e-05 .
    2021-08-15 12:34:23 [INFO]	[TRAIN] Epoch=14/270, Step=1/45, loss=18.228678, lr=7.3e-05, time_each_step=0.64s, eta=2:26:5
    2021-08-15 12:34:25 [INFO]	[TRAIN] Epoch=14/270, Step=3/45, loss=17.909086, lr=7.3e-05, time_each_step=0.7s, eta=2:26:18
    2021-08-15 12:34:27 [INFO]	[TRAIN] Epoch=14/270, Step=5/45, loss=12.583206, lr=7.4e-05, time_each_step=0.74s, eta=2:26:24
    2021-08-15 12:34:28 [INFO]	[TRAIN] Epoch=14/270, Step=7/45, loss=20.06926, lr=7.4e-05, time_each_step=0.76s, eta=2:26:28
    2021-08-15 12:34:29 [INFO]	[TRAIN] Epoch=14/270, Step=9/45, loss=14.272425, lr=7.4e-05, time_each_step=0.79s, eta=2:26:32
    2021-08-15 12:34:31 [INFO]	[TRAIN] Epoch=14/270, Step=11/45, loss=16.835707, lr=7.4e-05, time_each_step=0.84s, eta=2:26:41
    2021-08-15 12:34:32 [INFO]	[TRAIN] Epoch=14/270, Step=13/45, loss=13.840749, lr=7.5e-05, time_each_step=0.88s, eta=2:26:48
    2021-08-15 12:34:34 [INFO]	[TRAIN] Epoch=14/270, Step=15/45, loss=20.698256, lr=7.5e-05, time_each_step=0.91s, eta=2:26:52
    2021-08-15 12:34:34 [INFO]	[TRAIN] Epoch=14/270, Step=17/45, loss=14.105371, lr=7.5e-05, time_each_step=0.93s, eta=2:26:55
    2021-08-15 12:34:36 [INFO]	[TRAIN] Epoch=14/270, Step=19/45, loss=10.906362, lr=7.5e-05, time_each_step=0.97s, eta=2:27:1
    2021-08-15 12:34:37 [INFO]	[TRAIN] Epoch=14/270, Step=21/45, loss=14.951776, lr=7.6e-05, time_each_step=0.71s, eta=2:26:5
    2021-08-15 12:34:38 [INFO]	[TRAIN] Epoch=14/270, Step=23/45, loss=18.808344, lr=7.6e-05, time_each_step=0.64s, eta=2:25:50
    2021-08-15 12:34:39 [INFO]	[TRAIN] Epoch=14/270, Step=25/45, loss=17.116741, lr=7.6e-05, time_each_step=0.62s, eta=2:25:45
    2021-08-15 12:34:40 [INFO]	[TRAIN] Epoch=14/270, Step=27/45, loss=14.331366, lr=7.6e-05, time_each_step=0.61s, eta=2:25:42
    2021-08-15 12:34:42 [INFO]	[TRAIN] Epoch=14/270, Step=29/45, loss=15.634377, lr=7.7e-05, time_each_step=0.61s, eta=2:25:40
    2021-08-15 12:34:43 [INFO]	[TRAIN] Epoch=14/270, Step=31/45, loss=19.891068, lr=7.7e-05, time_each_step=0.59s, eta=2:25:37
    2021-08-15 12:34:44 [INFO]	[TRAIN] Epoch=14/270, Step=33/45, loss=14.376038, lr=7.7e-05, time_each_step=0.57s, eta=2:25:31
    2021-08-15 12:34:44 [INFO]	[TRAIN] Epoch=14/270, Step=35/45, loss=12.830074, lr=7.7e-05, time_each_step=0.53s, eta=2:25:22
    2021-08-15 12:34:45 [INFO]	[TRAIN] Epoch=14/270, Step=37/45, loss=16.222347, lr=7.8e-05, time_each_step=0.52s, eta=2:25:18
    2021-08-15 12:34:45 [INFO]	[TRAIN] Epoch=14/270, Step=39/45, loss=12.603162, lr=7.8e-05, time_each_step=0.49s, eta=2:25:13
    2021-08-15 12:34:46 [INFO]	[TRAIN] Epoch=14/270, Step=41/45, loss=17.849838, lr=7.8e-05, time_each_step=0.45s, eta=2:25:3
    2021-08-15 12:34:46 [INFO]	[TRAIN] Epoch=14/270, Step=43/45, loss=14.644626, lr=7.8e-05, time_each_step=0.41s, eta=2:24:56
    2021-08-15 12:34:47 [INFO]	[TRAIN] Epoch=14/270, Step=45/45, loss=14.923976, lr=7.9e-05, time_each_step=0.37s, eta=2:24:48
    2021-08-15 12:34:47 [INFO]	[TRAIN] Epoch 14 finished, loss=16.073957, lr=7.6e-05 .
    2021-08-15 12:34:55 [INFO]	[TRAIN] Epoch=15/270, Step=2/45, loss=17.623569, lr=7.9e-05, time_each_step=0.74s, eta=2:10:53
    2021-08-15 12:34:57 [INFO]	[TRAIN] Epoch=15/270, Step=4/45, loss=16.061949, lr=7.9e-05, time_each_step=0.78s, eta=2:10:59
    2021-08-15 12:34:59 [INFO]	[TRAIN] Epoch=15/270, Step=6/45, loss=16.783518, lr=7.9e-05, time_each_step=0.81s, eta=2:11:4
    2021-08-15 12:35:00 [INFO]	[TRAIN] Epoch=15/270, Step=8/45, loss=15.540409, lr=8e-05, time_each_step=0.82s, eta=2:11:6
    2021-08-15 12:35:02 [INFO]	[TRAIN] Epoch=15/270, Step=10/45, loss=11.587591, lr=8e-05, time_each_step=0.89s, eta=2:11:19
    2021-08-15 12:35:03 [INFO]	[TRAIN] Epoch=15/270, Step=12/45, loss=14.617434, lr=8e-05, time_each_step=0.93s, eta=2:11:26
    2021-08-15 12:35:05 [INFO]	[TRAIN] Epoch=15/270, Step=14/45, loss=13.340837, lr=8e-05, time_each_step=0.97s, eta=2:11:33
    2021-08-15 12:35:06 [INFO]	[TRAIN] Epoch=15/270, Step=16/45, loss=14.603296, lr=8.1e-05, time_each_step=1.01s, eta=2:11:40
    2021-08-15 12:35:07 [INFO]	[TRAIN] Epoch=15/270, Step=18/45, loss=16.845142, lr=8.1e-05, time_each_step=1.05s, eta=2:11:44
    2021-08-15 12:35:08 [INFO]	[TRAIN] Epoch=15/270, Step=20/45, loss=12.143922, lr=8.1e-05, time_each_step=1.08s, eta=2:11:49
    2021-08-15 12:35:09 [INFO]	[TRAIN] Epoch=15/270, Step=22/45, loss=12.157655, lr=8.1e-05, time_each_step=0.7s, eta=2:10:29
    2021-08-15 12:35:11 [INFO]	[TRAIN] Epoch=15/270, Step=24/45, loss=13.556926, lr=8.2e-05, time_each_step=0.68s, eta=2:10:24
    2021-08-15 12:35:12 [INFO]	[TRAIN] Epoch=15/270, Step=26/45, loss=12.080097, lr=8.2e-05, time_each_step=0.64s, eta=2:10:15
    2021-08-15 12:35:13 [INFO]	[TRAIN] Epoch=15/270, Step=28/45, loss=16.267637, lr=8.2e-05, time_each_step=0.63s, eta=2:10:12
    2021-08-15 12:35:14 [INFO]	[TRAIN] Epoch=15/270, Step=30/45, loss=16.24032, lr=8.2e-05, time_each_step=0.59s, eta=2:10:3
    2021-08-15 12:35:15 [INFO]	[TRAIN] Epoch=15/270, Step=32/45, loss=18.308046, lr=8.3e-05, time_each_step=0.57s, eta=2:9:56
    2021-08-15 12:35:16 [INFO]	[TRAIN] Epoch=15/270, Step=34/45, loss=14.867095, lr=8.3e-05, time_each_step=0.53s, eta=2:9:49
    2021-08-15 12:35:16 [INFO]	[TRAIN] Epoch=15/270, Step=36/45, loss=14.951165, lr=8.3e-05, time_each_step=0.5s, eta=2:9:40
    2021-08-15 12:35:16 [INFO]	[TRAIN] Epoch=15/270, Step=38/45, loss=14.15453, lr=8.3e-05, time_each_step=0.47s, eta=2:9:35
    2021-08-15 12:35:17 [INFO]	[TRAIN] Epoch=15/270, Step=40/45, loss=18.635271, lr=8.4e-05, time_each_step=0.44s, eta=2:9:27
    2021-08-15 12:35:18 [INFO]	[TRAIN] Epoch=15/270, Step=42/45, loss=13.449648, lr=8.4e-05, time_each_step=0.42s, eta=2:9:23
    2021-08-15 12:35:18 [INFO]	[TRAIN] Epoch=15/270, Step=44/45, loss=18.813345, lr=8.4e-05, time_each_step=0.37s, eta=2:9:14
    2021-08-15 12:35:18 [INFO]	[TRAIN] Epoch 15 finished, loss=15.495304, lr=8.2e-05 .
    2021-08-15 12:35:26 [INFO]	[TRAIN] Epoch=16/270, Step=1/45, loss=19.088673, lr=8.4e-05, time_each_step=0.71s, eta=2:17:21
    2021-08-15 12:35:28 [INFO]	[TRAIN] Epoch=16/270, Step=3/45, loss=18.963631, lr=8.5e-05, time_each_step=0.74s, eta=2:17:27
    2021-08-15 12:35:30 [INFO]	[TRAIN] Epoch=16/270, Step=5/45, loss=21.618956, lr=8.5e-05, time_each_step=0.79s, eta=2:17:36
    2021-08-15 12:35:31 [INFO]	[TRAIN] Epoch=16/270, Step=7/45, loss=13.864916, lr=8.5e-05, time_each_step=0.82s, eta=2:17:41
    2021-08-15 12:35:33 [INFO]	[TRAIN] Epoch=16/270, Step=9/45, loss=15.475196, lr=8.5e-05, time_each_step=0.88s, eta=2:17:53
    2021-08-15 12:35:34 [INFO]	[TRAIN] Epoch=16/270, Step=11/45, loss=19.871029, lr=8.6e-05, time_each_step=0.92s, eta=2:18:1
    2021-08-15 12:35:36 [INFO]	[TRAIN] Epoch=16/270, Step=13/45, loss=16.809181, lr=8.6e-05, time_each_step=0.98s, eta=2:18:11
    2021-08-15 12:35:37 [INFO]	[TRAIN] Epoch=16/270, Step=15/45, loss=16.733545, lr=8.6e-05, time_each_step=1.02s, eta=2:18:18
    2021-08-15 12:35:38 [INFO]	[TRAIN] Epoch=16/270, Step=17/45, loss=11.426452, lr=8.6e-05, time_each_step=1.03s, eta=2:18:19
    2021-08-15 12:35:40 [INFO]	[TRAIN] Epoch=16/270, Step=19/45, loss=16.226261, lr=8.7e-05, time_each_step=1.07s, eta=2:18:24
    2021-08-15 12:35:41 [INFO]	[TRAIN] Epoch=16/270, Step=21/45, loss=17.207129, lr=8.7e-05, time_each_step=0.75s, eta=2:17:15
    2021-08-15 12:35:42 [INFO]	[TRAIN] Epoch=16/270, Step=23/45, loss=13.186077, lr=8.7e-05, time_each_step=0.71s, eta=2:17:7
    2021-08-15 12:35:43 [INFO]	[TRAIN] Epoch=16/270, Step=25/45, loss=11.802549, lr=8.7e-05, time_each_step=0.67s, eta=2:16:56
    2021-08-15 12:35:44 [INFO]	[TRAIN] Epoch=16/270, Step=27/45, loss=13.231317, lr=8.8e-05, time_each_step=0.67s, eta=2:16:55
    2021-08-15 12:35:46 [INFO]	[TRAIN] Epoch=16/270, Step=29/45, loss=12.271926, lr=8.8e-05, time_each_step=0.64s, eta=2:16:47
    2021-08-15 12:35:47 [INFO]	[TRAIN] Epoch=16/270, Step=31/45, loss=16.989082, lr=8.8e-05, time_each_step=0.6s, eta=2:16:40
    2021-08-15 12:35:48 [INFO]	[TRAIN] Epoch=16/270, Step=33/45, loss=16.377689, lr=8.8e-05, time_each_step=0.57s, eta=2:16:33
    2021-08-15 12:35:48 [INFO]	[TRAIN] Epoch=16/270, Step=35/45, loss=13.656185, lr=8.9e-05, time_each_step=0.53s, eta=2:16:24
    2021-08-15 12:35:48 [INFO]	[TRAIN] Epoch=16/270, Step=37/45, loss=20.612612, lr=8.9e-05, time_each_step=0.5s, eta=2:16:17
    2021-08-15 12:35:49 [INFO]	[TRAIN] Epoch=16/270, Step=39/45, loss=12.839915, lr=8.9e-05, time_each_step=0.47s, eta=2:16:10
    2021-08-15 12:35:49 [INFO]	[TRAIN] Epoch=16/270, Step=41/45, loss=15.518668, lr=8.9e-05, time_each_step=0.44s, eta=2:16:3
    2021-08-15 12:35:50 [INFO]	[TRAIN] Epoch=16/270, Step=43/45, loss=14.149256, lr=9e-05, time_each_step=0.4s, eta=2:15:56
    2021-08-15 12:35:50 [INFO]	[TRAIN] Epoch=16/270, Step=45/45, loss=21.685968, lr=9e-05, time_each_step=0.37s, eta=2:15:49
    2021-08-15 12:35:50 [INFO]	[TRAIN] Epoch 16 finished, loss=15.87251, lr=8.7e-05 .
    2021-08-15 12:35:58 [INFO]	[TRAIN] Epoch=17/270, Step=2/45, loss=13.031922, lr=9e-05, time_each_step=0.66s, eta=2:17:21
    2021-08-15 12:36:00 [INFO]	[TRAIN] Epoch=17/270, Step=4/45, loss=37.948009, lr=9e-05, time_each_step=0.7s, eta=2:17:29
    2021-08-15 12:36:02 [INFO]	[TRAIN] Epoch=17/270, Step=6/45, loss=15.09861, lr=9.1e-05, time_each_step=0.78s, eta=2:17:45
    2021-08-15 12:36:03 [INFO]	[TRAIN] Epoch=17/270, Step=8/45, loss=15.668639, lr=9.1e-05, time_each_step=0.79s, eta=2:17:46
    2021-08-15 12:36:04 [INFO]	[TRAIN] Epoch=17/270, Step=10/45, loss=11.714021, lr=9.1e-05, time_each_step=0.81s, eta=2:17:48
    2021-08-15 12:36:05 [INFO]	[TRAIN] Epoch=17/270, Step=12/45, loss=12.454744, lr=9.1e-05, time_each_step=0.85s, eta=2:17:56
    2021-08-15 12:36:07 [INFO]	[TRAIN] Epoch=17/270, Step=14/45, loss=12.90599, lr=9.2e-05, time_each_step=0.88s, eta=2:18:1
    2021-08-15 12:36:08 [INFO]	[TRAIN] Epoch=17/270, Step=16/45, loss=19.899872, lr=9.2e-05, time_each_step=0.9s, eta=2:18:4
    2021-08-15 12:36:09 [INFO]	[TRAIN] Epoch=17/270, Step=18/45, loss=10.80677, lr=9.2e-05, time_each_step=0.95s, eta=2:18:12
    2021-08-15 12:36:10 [INFO]	[TRAIN] Epoch=17/270, Step=20/45, loss=14.364873, lr=9.2e-05, time_each_step=1.0s, eta=2:18:19
    2021-08-15 12:36:12 [INFO]	[TRAIN] Epoch=17/270, Step=22/45, loss=11.242459, lr=9.3e-05, time_each_step=0.71s, eta=2:17:18
    2021-08-15 12:36:13 [INFO]	[TRAIN] Epoch=17/270, Step=24/45, loss=16.796219, lr=9.3e-05, time_each_step=0.65s, eta=2:17:6
    2021-08-15 12:36:14 [INFO]	[TRAIN] Epoch=17/270, Step=26/45, loss=14.137141, lr=9.3e-05, time_each_step=0.6s, eta=2:16:53
    2021-08-15 12:36:15 [INFO]	[TRAIN] Epoch=17/270, Step=28/45, loss=11.469574, lr=9.3e-05, time_each_step=0.58s, eta=2:16:48
    2021-08-15 12:36:16 [INFO]	[TRAIN] Epoch=17/270, Step=30/45, loss=14.891869, lr=9.4e-05, time_each_step=0.6s, eta=2:16:51
    2021-08-15 12:36:17 [INFO]	[TRAIN] Epoch=17/270, Step=32/45, loss=12.405824, lr=9.4e-05, time_each_step=0.57s, eta=2:16:44
    2021-08-15 12:36:18 [INFO]	[TRAIN] Epoch=17/270, Step=34/45, loss=15.745978, lr=9.4e-05, time_each_step=0.55s, eta=2:16:40
    2021-08-15 12:36:18 [INFO]	[TRAIN] Epoch=17/270, Step=36/45, loss=15.077756, lr=9.4e-05, time_each_step=0.53s, eta=2:16:35
    2021-08-15 12:36:19 [INFO]	[TRAIN] Epoch=17/270, Step=38/45, loss=16.050888, lr=9.5e-05, time_each_step=0.49s, eta=2:16:25
    2021-08-15 12:36:19 [INFO]	[TRAIN] Epoch=17/270, Step=40/45, loss=13.674162, lr=9.5e-05, time_each_step=0.44s, eta=2:16:16
    2021-08-15 12:36:20 [INFO]	[TRAIN] Epoch=17/270, Step=42/45, loss=15.365814, lr=9.5e-05, time_each_step=0.41s, eta=2:16:8
    2021-08-15 12:36:20 [INFO]	[TRAIN] Epoch=17/270, Step=44/45, loss=13.637261, lr=9.5e-05, time_each_step=0.37s, eta=2:16:2
    2021-08-15 12:36:21 [INFO]	[TRAIN] Epoch 17 finished, loss=14.940491, lr=9.3e-05 .
    2021-08-15 12:36:27 [INFO]	[TRAIN] Epoch=18/270, Step=1/45, loss=14.365431, lr=9.6e-05, time_each_step=0.65s, eta=2:9:20
    2021-08-15 12:36:29 [INFO]	[TRAIN] Epoch=18/270, Step=3/45, loss=7.988242, lr=9.6e-05, time_each_step=0.68s, eta=2:9:27
    2021-08-15 12:36:30 [INFO]	[TRAIN] Epoch=18/270, Step=5/45, loss=12.949059, lr=9.6e-05, time_each_step=0.7s, eta=2:9:29
    2021-08-15 12:36:32 [INFO]	[TRAIN] Epoch=18/270, Step=7/45, loss=9.423842, lr=9.6e-05, time_each_step=0.73s, eta=2:9:35
    2021-08-15 12:36:33 [INFO]	[TRAIN] Epoch=18/270, Step=9/45, loss=11.593818, lr=9.7e-05, time_each_step=0.77s, eta=2:9:42
    2021-08-15 12:36:34 [INFO]	[TRAIN] Epoch=18/270, Step=11/45, loss=12.426253, lr=9.7e-05, time_each_step=0.8s, eta=2:9:47
    2021-08-15 12:36:36 [INFO]	[TRAIN] Epoch=18/270, Step=13/45, loss=10.321693, lr=9.7e-05, time_each_step=0.85s, eta=2:9:56
    2021-08-15 12:36:37 [INFO]	[TRAIN] Epoch=18/270, Step=15/45, loss=17.084335, lr=9.7e-05, time_each_step=0.89s, eta=2:10:2
    2021-08-15 12:36:38 [INFO]	[TRAIN] Epoch=18/270, Step=17/45, loss=14.185034, lr=9.8e-05, time_each_step=0.93s, eta=2:10:9
    2021-08-15 12:36:40 [INFO]	[TRAIN] Epoch=18/270, Step=19/45, loss=13.105035, lr=9.8e-05, time_each_step=1.0s, eta=2:10:22
    2021-08-15 12:36:41 [INFO]	[TRAIN] Epoch=18/270, Step=21/45, loss=12.103322, lr=9.8e-05, time_each_step=0.72s, eta=2:9:22
    2021-08-15 12:36:43 [INFO]	[TRAIN] Epoch=18/270, Step=23/45, loss=13.083316, lr=9.8e-05, time_each_step=0.7s, eta=2:9:17
    2021-08-15 12:36:44 [INFO]	[TRAIN] Epoch=18/270, Step=25/45, loss=16.68194, lr=9.9e-05, time_each_step=0.68s, eta=2:9:11
    2021-08-15 12:36:45 [INFO]	[TRAIN] Epoch=18/270, Step=27/45, loss=10.297939, lr=9.9e-05, time_each_step=0.67s, eta=2:9:9
    2021-08-15 12:36:46 [INFO]	[TRAIN] Epoch=18/270, Step=29/45, loss=15.476871, lr=9.9e-05, time_each_step=0.65s, eta=2:9:3
    2021-08-15 12:36:47 [INFO]	[TRAIN] Epoch=18/270, Step=31/45, loss=13.172777, lr=9.9e-05, time_each_step=0.64s, eta=2:9:0
    2021-08-15 12:36:48 [INFO]	[TRAIN] Epoch=18/270, Step=33/45, loss=16.064184, lr=0.0001, time_each_step=0.61s, eta=2:8:53
    2021-08-15 12:36:49 [INFO]	[TRAIN] Epoch=18/270, Step=35/45, loss=11.61609, lr=0.0001, time_each_step=0.58s, eta=2:8:46
    2021-08-15 12:36:49 [INFO]	[TRAIN] Epoch=18/270, Step=37/45, loss=11.650252, lr=0.0001, time_each_step=0.53s, eta=2:8:35
    2021-08-15 12:36:50 [INFO]	[TRAIN] Epoch=18/270, Step=39/45, loss=12.399238, lr=0.0001, time_each_step=0.46s, eta=2:8:21
    2021-08-15 12:36:50 [INFO]	[TRAIN] Epoch=18/270, Step=41/45, loss=20.801424, lr=0.000101, time_each_step=0.44s, eta=2:8:17
    2021-08-15 12:36:51 [INFO]	[TRAIN] Epoch=18/270, Step=43/45, loss=17.020046, lr=0.000101, time_each_step=0.4s, eta=2:8:7
    2021-08-15 12:36:51 [INFO]	[TRAIN] Epoch=18/270, Step=45/45, loss=18.508171, lr=0.000101, time_each_step=0.37s, eta=2:8:2
    2021-08-15 12:36:51 [INFO]	[TRAIN] Epoch 18 finished, loss=13.982823, lr=9.8e-05 .
    2021-08-15 12:36:58 [INFO]	[TRAIN] Epoch=19/270, Step=2/45, loss=17.64703, lr=0.000101, time_each_step=0.67s, eta=2:10:30
    2021-08-15 12:37:01 [INFO]	[TRAIN] Epoch=19/270, Step=4/45, loss=16.179489, lr=0.000102, time_each_step=0.72s, eta=2:10:41
    2021-08-15 12:37:02 [INFO]	[TRAIN] Epoch=19/270, Step=6/45, loss=16.131752, lr=0.000102, time_each_step=0.76s, eta=2:10:48
    2021-08-15 12:37:04 [INFO]	[TRAIN] Epoch=19/270, Step=8/45, loss=11.971616, lr=0.000102, time_each_step=0.8s, eta=2:10:54
    2021-08-15 12:37:06 [INFO]	[TRAIN] Epoch=19/270, Step=10/45, loss=17.960793, lr=0.000102, time_each_step=0.85s, eta=2:11:3
    2021-08-15 12:37:07 [INFO]	[TRAIN] Epoch=19/270, Step=12/45, loss=14.777832, lr=0.000103, time_each_step=0.87s, eta=2:11:7
    2021-08-15 12:37:08 [INFO]	[TRAIN] Epoch=19/270, Step=14/45, loss=19.370653, lr=0.000103, time_each_step=0.91s, eta=2:11:13
    2021-08-15 12:37:09 [INFO]	[TRAIN] Epoch=19/270, Step=16/45, loss=10.694462, lr=0.000103, time_each_step=0.94s, eta=2:11:18
    2021-08-15 12:37:10 [INFO]	[TRAIN] Epoch=19/270, Step=18/45, loss=16.332352, lr=0.000103, time_each_step=0.98s, eta=2:11:25
    2021-08-15 12:37:11 [INFO]	[TRAIN] Epoch=19/270, Step=20/45, loss=12.078723, lr=0.000104, time_each_step=1.0s, eta=2:11:27
    2021-08-15 12:37:13 [INFO]	[TRAIN] Epoch=19/270, Step=22/45, loss=13.828105, lr=0.000104, time_each_step=0.71s, eta=2:10:26
    2021-08-15 12:37:14 [INFO]	[TRAIN] Epoch=19/270, Step=24/45, loss=15.368047, lr=0.000104, time_each_step=0.66s, eta=2:10:13
    2021-08-15 12:37:15 [INFO]	[TRAIN] Epoch=19/270, Step=26/45, loss=9.93207, lr=0.000104, time_each_step=0.62s, eta=2:10:3
    2021-08-15 12:37:16 [INFO]	[TRAIN] Epoch=19/270, Step=28/45, loss=14.170019, lr=0.000105, time_each_step=0.6s, eta=2:9:59
    2021-08-15 12:37:17 [INFO]	[TRAIN] Epoch=19/270, Step=30/45, loss=13.736856, lr=0.000105, time_each_step=0.56s, eta=2:9:50
    2021-08-15 12:37:18 [INFO]	[TRAIN] Epoch=19/270, Step=32/45, loss=15.566101, lr=0.000105, time_each_step=0.56s, eta=2:9:48
    2021-08-15 12:37:19 [INFO]	[TRAIN] Epoch=19/270, Step=34/45, loss=11.892754, lr=0.000105, time_each_step=0.54s, eta=2:9:43
    2021-08-15 12:37:19 [INFO]	[TRAIN] Epoch=19/270, Step=36/45, loss=15.568937, lr=0.000106, time_each_step=0.5s, eta=2:9:35
    2021-08-15 12:37:20 [INFO]	[TRAIN] Epoch=19/270, Step=38/45, loss=19.228912, lr=0.000106, time_each_step=0.46s, eta=2:9:27
    2021-08-15 12:37:20 [INFO]	[TRAIN] Epoch=19/270, Step=40/45, loss=19.264181, lr=0.000106, time_each_step=0.44s, eta=2:9:22
    2021-08-15 12:37:21 [INFO]	[TRAIN] Epoch=19/270, Step=42/45, loss=13.254473, lr=0.000106, time_each_step=0.39s, eta=2:9:12
    2021-08-15 12:37:21 [INFO]	[TRAIN] Epoch=19/270, Step=44/45, loss=11.931118, lr=0.000107, time_each_step=0.37s, eta=2:9:7
    2021-08-15 12:37:21 [INFO]	[TRAIN] Epoch 19 finished, loss=15.231544, lr=0.000104 .
    2021-08-15 12:37:35 [INFO]	[TRAIN] Epoch=20/270, Step=1/45, loss=13.242746, lr=0.000107, time_each_step=1.02s, eta=2:9:3
    2021-08-15 12:37:37 [INFO]	[TRAIN] Epoch=20/270, Step=3/45, loss=14.188869, lr=0.000107, time_each_step=1.07s, eta=2:9:14
    2021-08-15 12:37:39 [INFO]	[TRAIN] Epoch=20/270, Step=5/45, loss=14.988261, lr=0.000107, time_each_step=1.12s, eta=2:9:23
    2021-08-15 12:37:41 [INFO]	[TRAIN] Epoch=20/270, Step=7/45, loss=10.730973, lr=0.000108, time_each_step=1.16s, eta=2:9:29
    2021-08-15 12:37:42 [INFO]	[TRAIN] Epoch=20/270, Step=9/45, loss=11.940995, lr=0.000108, time_each_step=1.19s, eta=2:9:33
    2021-08-15 12:37:44 [INFO]	[TRAIN] Epoch=20/270, Step=11/45, loss=12.755896, lr=0.000108, time_each_step=1.25s, eta=2:9:43
    2021-08-15 12:37:45 [INFO]	[TRAIN] Epoch=20/270, Step=13/45, loss=12.284646, lr=0.000108, time_each_step=1.28s, eta=2:9:47
    2021-08-15 12:37:46 [INFO]	[TRAIN] Epoch=20/270, Step=15/45, loss=13.484276, lr=0.000109, time_each_step=1.3s, eta=2:9:49
    2021-08-15 12:37:47 [INFO]	[TRAIN] Epoch=20/270, Step=17/45, loss=13.827806, lr=0.000109, time_each_step=1.32s, eta=2:9:51
    2021-08-15 12:37:48 [INFO]	[TRAIN] Epoch=20/270, Step=19/45, loss=13.850746, lr=0.000109, time_each_step=1.35s, eta=2:9:54
    2021-08-15 12:37:49 [INFO]	[TRAIN] Epoch=20/270, Step=21/45, loss=12.960424, lr=0.000109, time_each_step=0.71s, eta=2:7:40
    2021-08-15 12:37:50 [INFO]	[TRAIN] Epoch=20/270, Step=23/45, loss=17.440674, lr=0.00011, time_each_step=0.65s, eta=2:7:25
    2021-08-15 12:37:52 [INFO]	[TRAIN] Epoch=20/270, Step=25/45, loss=13.225471, lr=0.00011, time_each_step=0.62s, eta=2:7:17
    2021-08-15 12:37:53 [INFO]	[TRAIN] Epoch=20/270, Step=27/45, loss=12.199125, lr=0.00011, time_each_step=0.59s, eta=2:7:11
    2021-08-15 12:37:54 [INFO]	[TRAIN] Epoch=20/270, Step=29/45, loss=12.88834, lr=0.00011, time_each_step=0.57s, eta=2:7:7
    2021-08-15 12:37:55 [INFO]	[TRAIN] Epoch=20/270, Step=31/45, loss=12.861481, lr=0.000111, time_each_step=0.54s, eta=2:7:0
    2021-08-15 12:37:56 [INFO]	[TRAIN] Epoch=20/270, Step=33/45, loss=17.303661, lr=0.000111, time_each_step=0.53s, eta=2:6:56
    2021-08-15 12:37:56 [INFO]	[TRAIN] Epoch=20/270, Step=35/45, loss=19.606129, lr=0.000111, time_each_step=0.51s, eta=2:6:50
    2021-08-15 12:37:57 [INFO]	[TRAIN] Epoch=20/270, Step=37/45, loss=21.194305, lr=0.000111, time_each_step=0.49s, eta=2:6:46
    2021-08-15 12:37:57 [INFO]	[TRAIN] Epoch=20/270, Step=39/45, loss=13.799168, lr=0.000112, time_each_step=0.46s, eta=2:6:40
    2021-08-15 12:37:58 [INFO]	[TRAIN] Epoch=20/270, Step=41/45, loss=15.126862, lr=0.000112, time_each_step=0.44s, eta=2:6:35
    2021-08-15 12:37:58 [INFO]	[TRAIN] Epoch=20/270, Step=43/45, loss=10.906478, lr=0.000112, time_each_step=0.41s, eta=2:6:28
    2021-08-15 12:37:59 [INFO]	[TRAIN] Epoch=20/270, Step=45/45, loss=15.353392, lr=0.000112, time_each_step=0.37s, eta=2:6:21
    2021-08-15 12:37:59 [INFO]	[TRAIN] Epoch 20 finished, loss=14.29339, lr=0.00011 .
    2021-08-15 12:37:59 [INFO]	Start to evaluating(total_samples=104, total_steps=13)...


      0%|          | 0/13 [00:09<?, ?it/s]



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-8-b5a65646add1> in <module>
         14     lr_decay_epochs=[210, 240],
         15     lr_decay_gamma=0.1,
    ---> 16     save_dir='output/yolov3_mobilenetv1')
    

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/models/ppyolo.py in train(self, num_epochs, train_dataset, train_batch_size, eval_dataset, save_interval_epochs, log_interval_steps, save_dir, pretrain_weights, optimizer, learning_rate, warmup_steps, warmup_start_lr, lr_decay_epochs, lr_decay_gamma, metric, use_vdl, sensitivities_file, eval_metric_loss, early_stop, early_stop_patience, resume_checkpoint, use_ema, ema_decay)
        364             use_vdl=use_vdl,
        365             early_stop=early_stop,
    --> 366             early_stop_patience=early_stop_patience)
        367 
        368     def evaluate(self,


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/models/base.py in train_loop(self, num_epochs, train_dataset, train_batch_size, eval_dataset, save_interval_epochs, log_interval_steps, save_dir, use_vdl, early_stop, early_stop_patience)
        579                         batch_size=eval_batch_size,
        580                         epoch_id=i + 1,
    --> 581                         return_details=True)
        582                     logging.info('[EVAL] Finished, Epoch={}, {} .'.format(
        583                         i + 1, dict2str(self.eval_metrics)))


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/models/ppyolo.py in evaluate(self, eval_dataset, batch_size, epoch_id, metric, return_details)
        429                     feed=[feed_data],
        430                     fetch_list=list(self.test_outputs.values()),
    --> 431                     return_numpy=False)
        432             res = {
        433                 'bbox': (np.array(outputs[0]),


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache, return_merged, use_prune)
       1108                 return_merged=return_merged)
       1109         except Exception as e:
    -> 1110             six.reraise(*sys.exc_info())
       1111 
       1112     def _run_impl(self, program, feed, fetch_list, feed_var_name,


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/six.py in reraise(tp, value, tb)
        701             if value.__traceback__ is not tb:
        702                 raise value.with_traceback(tb)
    --> 703             raise value
        704         finally:
        705             value = None


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/executor.py in run(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache, return_merged, use_prune)
       1106                 use_program_cache=use_program_cache,
       1107                 use_prune=use_prune,
    -> 1108                 return_merged=return_merged)
       1109         except Exception as e:
       1110             six.reraise(*sys.exc_info())


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/executor.py in _run_impl(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache, return_merged, use_prune)
       1237                 scope=scope,
       1238                 return_numpy=return_numpy,
    -> 1239                 use_program_cache=use_program_cache)
       1240 
       1241         program._compile(scope, self.place)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/executor.py in _run_program(self, program, feed, fetch_list, feed_var_name, fetch_var_name, scope, return_numpy, use_program_cache)
       1327         if not use_program_cache:
       1328             self._default_executor.run(program.desc, scope, 0, True, True,
    -> 1329                                        [fetch_var_name])
       1330         else:
       1331             self._default_executor.run_prepared_ctx(ctx, scope, False, False,


    ValueError: In user code:
    
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/runpy.py", line 193, in _run_module_as_main
          "__main__", mod_spec)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/runpy.py", line 85, in _run_code
          exec(code, run_globals)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py", line 16, in <module>
          app.launch_new_instance()
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/traitlets/config/application.py", line 664, in launch_instance
          app.start()
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel/kernelapp.py", line 505, in start
          self.io_loop.start()
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/tornado/platform/asyncio.py", line 132, in start
          self.asyncio_loop.run_forever()
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/asyncio/base_events.py", line 534, in run_forever
          self._run_once()
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/asyncio/base_events.py", line 1771, in _run_once
          handle._run()
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/asyncio/events.py", line 88, in _run
          self._context.run(self._callback, *self._args)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/tornado/ioloop.py", line 758, in _run_callback
          ret = callback()
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/tornado/stack_context.py", line 300, in null_wrapper
          return fn(*args, **kwargs)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/tornado/gen.py", line 1233, in inner
          self.run()
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/tornado/gen.py", line 1147, in run
          yielded = self.gen.send(value)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel/kernelbase.py", line 357, in process_one
          yield gen.maybe_future(dispatch(*args))
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/tornado/gen.py", line 326, in wrapper
          yielded = next(result)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel/kernelbase.py", line 267, in dispatch_shell
          yield gen.maybe_future(handler(stream, idents, msg))
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/tornado/gen.py", line 326, in wrapper
          yielded = next(result)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel/kernelbase.py", line 534, in execute_request
          user_expressions, allow_stdin,
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/tornado/gen.py", line 326, in wrapper
          yielded = next(result)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel/ipkernel.py", line 294, in do_execute
          res = shell.run_cell(code, store_history=store_history, silent=silent)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel/zmqshell.py", line 536, in run_cell
          return super(ZMQInteractiveShell, self).run_cell(*args, **kwargs)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 2817, in run_cell
          raw_cell, store_history, silent, shell_futures)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 2843, in _run_cell
          return runner(coro)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/IPython/core/async_helpers.py", line 67, in _pseudo_sync_runner
          coro.send(None)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3018, in run_cell_async
          interactivity=interactivity, compiler=compiler, result=result)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3189, in run_ast_nodes
          if (yield from self.run_code(code, result)):
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3265, in run_code
          exec(code_obj, self.user_global_ns, self.user_ns)
        File "<ipython-input-8-b5a65646add1>", line 16, in <module>
          save_dir='output/yolov3_mobilenetv1')
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/models/ppyolo.py", line 346, in train
          self.build_program()
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/models/base.py", line 114, in build_program
          mode='test')
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/models/ppyolo.py", line 175, in build_net
          model_out = model.build_net(inputs)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/yolo_v3.py", line 512, in build_net
          head_outputs = self._head(feats, self.mode == 'train')
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/yolo_v3.py", line 148, in _head
          name=self.prefix_name + "yolo_block.{}".format(i))
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/yolo_v3.py", line 335, in _detection_block
          conv = self._add_coord(conv, is_test=is_test)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/detection/yolo_v3.py", line 253, in _add_coord
          return fluid.layers.concat([input, x_range, y_range], axis=1)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/tensor.py", line 356, in concat
          type='concat', inputs=inputs, outputs={'Out': [out]}, attrs=attrs)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layer_helper.py", line 43, in append_op
          return self.main_program.current_block().append_op(*args, **kwargs)
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py", line 2942, in append_op
          attrs=kwargs.get("attrs", None))
        File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py", line 2014, in __init__
          for frame in traceback.extract_stack():
    
        InvalidArgumentError: The 2-th dimension of input[0] and input[2] is expected to be equal.But received input[0]'s shape = [8, 2048, 24, 42], input[2]'s shape = [8, 1, 42, 24].
          [Hint: Expected inputs_dims[0][j] == inputs_dims[i][j], but received inputs_dims[0][j]:24 != inputs_dims[i][j]:42.] (at /paddle/paddle/fluid/operators/concat_op.h:63)
          [operator < concat > error]



```python
!tar -xf /home/aistudio/output/yolov3_mobilenetv1/pretrain/ResNet50_vd_ssld_pretrained.tar
```

<font size ="5" color="red">写在最后</fort>

<font size ="3" >关于本次作业，嗯......，好像是跑通了，也好像没跑通，总之应该是参数设定还不足够完善。（感觉没跑通是因为最后结束出现了一些参数报错，但感觉跑通了，是最终也生成了压缩包，我还试着解压了一下），还有就是不知道再加啥参数了（看了看参考链接，将参数基本都用了，因为担心参数数值设置出错，基本还都用的默认参数）。参加这个活动，第一次接触，项目以后还需继续完善。

