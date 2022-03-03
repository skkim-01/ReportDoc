## KPFBERT

### [KPFBERT / kpfbert](https://github.com/KPFBERT/kpfbert)
### [Detail Document](https://github.com/KPFBERT/kpfbert/blob/main/BERT-MediaNavi.pdf)

### Test Summary
- GIT에서 제공하는 파일 다운로드
- 사용 방법에 따른 샘플 코드 작성 및 수정
- 결과 출력

### TODO
- 상기 설명 PDF 정독 및 이해
- GOOGLE BERT 분석
  - https://github.com/google-research/bert
  - https://www.tensorflow.org/text/tutorials/classify_text_with_bert

---

### Environment
```
- Windows 10 Pro
- Python 3.9.1
```

### Sample Code
- `main.py`
```python
from transformers import BertModel, BertTokenizer
import torch

print("### Start of Process ###")
print()
#model_name_or_path = "LOCAL_MODEL_PATH"  # Bert 바이너리가 포함된 디렉토리
strConfigPath = "D:\\pypro\\kpfbert\\"

model = BertModel.from_pretrained(strConfigPath, add_pooling_layer=False)
tokenizer = BertTokenizer.from_pretrained(strConfigPath)

strSampleText = "언론진흥재단 BERT 모델을 공개합니다.\n모델을 이용한 테스트 코드를 구현하였습니다."

print("### Sample Text ###")
print(strSampleText)
print()

strTokenizedText = tokenizer.tokenize(strSampleText)
print("### Tokenized Text ###")
print(strTokenizedText)
print()

encodedData = tokenizer(strSampleText)
print("### Encoded Token Data ###")
print(encodedData)
print()

model.eval()
pEncodedInput = tokenizer(strSampleText, return_tensors="pt")
result = model(**pEncodedInput, return_dict=False)
print("### Model Result ###")
print(result)
print()

print("### End of Process ###")
```

### Result
- Some weigths...: warning message / 설정을 통해 수정 가능한 것으로 보임

```bash
PS D:\pypro\tbert\tb> python .\main.py 
### start of main ###

Some weights of the model checkpoint at D:\pypro\kpfbert\ were not used when initializing BertModel: ['cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

### Sample Text ###
언론진흥재단 BERT 모델을 공개합니다.
모델을 이용한 테스트 코드를 구현하였습니다.

### Tokenized Text ###
['언론', '##진흥', '##재단', 'BE', '##RT', '모델', '##을', '공개', '##합니다', '.', '모델', '##을', '이용한', '테스트', '코드', '##를', '구현', '##하', '##였', '##습', '##니다', '.']

### Encoded Token Data ###
{'input_ids': [2, 7392, 24220, 16227, 28024, 21924, 7522, 4620, 7247, 15801, 518, 7522, 4620, 33598, 10670, 10297, 4630, 10160, 4554, 5406, 4935, 6808, 518, 3], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

### Model Result ###
(tensor([[[ 0.2225, -2.7142, -1.2773,  ..., -0.7680,  0.3923,  0.1205],
         [-0.0150, -0.2789, -0.6891,  ...,  1.9095,  0.2224,  0.6719],
         [-0.2935, -0.4841,  0.3588,  ...,  1.6318, -1.6325, -0.3881],
         ...,
         [-0.6542,  0.8549, -0.8495,  ...,  1.8794, -1.1778,  0.7892],
         [-0.0097, -0.7113,  0.0788,  ...,  1.0742, -0.1755,  0.5655],
         [ 0.2224, -2.7142, -1.2776,  ..., -0.7678,  0.3922,  0.1203]]],
       grad_fn=<NativeLayerNormBackward0>), None)

### End of Process ##
```