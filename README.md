# multi-label-classification
MONDEIQUE MAIN AI CODE : Multi-Label(Attribute) Image Classification at Handbag Image
<br></br>
## prerequsite
```
$ pip install -r requirements.txt
```
## How to run
- jupyter notebook 의 train.ipynb 파일을 이용하는 방법

- 자신이 직접 train.py를 짜 ```$ python train.py```로 실행하는 방법
<br></br>
## file

### python file


1.  const.py : learning rate나 batch size와 같은 constant 변수를 저장하는 python code

2.  data_loader.py : numpy 형태로 train_data와 test_data를 나누는 python code와 batch size의 형태로 애초에 초반에 pipeline을 만들어주는 python code

3.  data_utils.py : numpy 형태의 data와 tensor 형태의 data를 augment해주는 python code

4.  model.py : main multi-label image classification model python code로 branch가 뻗어져나와 feature extract 수행 및 selective loss

5.  ops.py : model을 설계하기 위한 layer operation python code

### jupyter notebook file

1.  train.ipynb : main training jupyter notebook file 
<br></br>
## data 

### data/bag_image

- local에 저장되어 있는 모든 handbag training / evaluation data image입니다. 이 repo에서는 용량 문제로 인한 example image 가 들어있습니다.

### data/bag_image_csv

- Amazon s3에 있는 cropped image id 와 matching 되는 filename 이 들어있는 csv

- 각 cropped image id에 매칭되는 category label 이 들어있는 csv

### data/training_bag.csv + data/make_final_csv

- 최종 training csv를 만들기 위한 jupyter notebook file 과 생성된 training csv
<br></br>
## error

1. cropped_image_id가 중복되어 두 번의 카테고리 라벨이 저장되는 경우 발생 : drop_duplicates()를 이용하여 해결
