# multi-label-classification
MONDEIQUE MAIN AI CODE : Multi-Label(Attribute) Image Classification at Handbag Image

## file

### python file

`1. const.py : learning rate나 batch size와 같은 constant 변수를 저장하는 python code`

`2. data_loader.py : numpy 형태로 train_data와 test_data를 나누는 python code와 batch size의 형태로 애초에 초반에 pipeline을 만들어주는 python code`

`3. data_utils.py : numpy 형태의 data와 tensor 형태의 data를 augment해주는 python code`

`4. model.py : main multi-label image classification model python code (loss, feature extraction)`

`5. ops.py : model을 설계하기 위한 layer operation python code`

<br></br>
### data/bag_image

local에 저장되어 있는 모든 handbag training / evaluation data image입니다. 이 repo에서는 용량 문제로 인한 example image 가 들어있습니다.

### data/bag_image_csv

각 handbag image에 대한 filename과 각 label이 달려있는 csv입니다. 