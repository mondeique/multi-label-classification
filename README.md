# multi-label-classification
MONDEIQUE MAIN AI CODE : Multi-Label(Attribute) Image Classification at Handbag Image
<br></br>
## prerequsite
```
$ pip install -r requirements.txt
```
## How to run
```
train.ipynb run cell
```

```
evaluation.ipynb run cell
```

```
test.ipynb run cell
```


## file

### python file

1.  const.py : learning rate나 image size, batch size와 같은 constant를 가지고 있는 python code

2.  data_loader.py : numpy 형태로 train_data와 test_data를 나누는 python code와 batch size의 형태로 애초에 초반에 pipeline을 만들어주는 python code

3.  data_utils.py : numpy 형태의 data와 tensor 형태의 data를 augment해주는 python code

4.  model.py : main multi-label image classification model python code로 branch가 뻗어져나와 feature extract 수행 및 mask에 따른 selective loss 설정

5.  ops.py : model을 설계하기 위한 layer operation python code

### jupyter notebook file (private repo 이므로 preview는 불가능) - have to change python file

1.  train.ipynb : training data에 대한 main training jupyter notebook code

2.  evaluate.ipynb : evlautaion data에 대한 evaluation jupyter notebook code

3.  test.ipynb : opencv로 test image에 있는 image를 불러와 미리 ```saver.restore(sess,)```에 load 된 model로 predict 하는 code
<br></br>
## data 

### data/bag_image

- local에 저장되어 있는 모든 handbag training / evaluation / test data image입니다. 이 repo에서는 용량 문제로 인한 example image 가 들어있습니다.

### data/bag_image_csv

- choco_categories_categories.csv :  각 cropped image id에 매칭되는 category label 이 들어있는 csv

- choco_categories_croppedimage.csv : Amazon s3에 있는 cropped image id 와 matching 되는 filename 이 들어있는 csv

### data/training_bag.csv + data/make_final_csv

- 최종 training csv를 만들기 위한 jupyter notebook file 과 생성된 training + evaluation csv (함께 있으며 나중에 data_loader 과정에서 나뉨)

<br></br>
## error

1. cropped_image_id가 중복되어 두 번의 카테고리 라벨이 저장되는 경우 발생 : drop_duplicates()를 이용하여 해결

2. numpy array - list 변환 시 RAM memory를 크게 잡아먹어 큰 속도의 감소 : pre-processing으로 cv2 과정에서 다시 list로 변환

3. ValueError: setting an array element with a sequence. -> array와 list의 혼용으로 인한 feed_dict에서의 shape 문제

4. ValueError: Cannot feed value of shape (16, 1) for Tensor 'Placeholder_1:0', which has shape '(?, 8)' : label tensor 선언시 size [BATCH SIZE, 8]로 선언했던 문제 -> reshape로 shape 맞춰줌

5. ValueError: Cannot feed value of shape (16, 1) for Tensor 'Placeholder_2:0', which has shape '(16,)' : np.reshape로 shape 맞춰줘야 했던 문제 -> reshape로 shape 맞춰줌
