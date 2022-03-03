## KPFSBERT

### [KPFBERT / kpfSBERT](https://github.com/KPFBERT/kpfSBERT)
### [Detail Document](https://github.com/KPFBERT/kpfSBERT/blob/main/kpfSBERT.ipynb)

### Test Summary
- Jupyter에 기록된 내용을 기반으로 코드 작성
- SBERT 모델 빌드: `build.py`
- 문장 유사성 추출 테스트: `test.py`

### TODO
- 모델에 대한 빌드와 학습에 대한 이해
- 문장 유사성 테스트 코드에 대한 이해와 활용 방안 모색

### Issues
- SBERT 모델을 빌드하고 학습하는데 약 30시간 소요
    - 외장 그래픽카드와의 인과 관계는 명확하지 않으나, CPU 사용 모니터링 결과 영향은 끼칠 수 있다고 판단
- 성능 관련 스크린샷은 별도 보유

---

### Environment
```
- Windows 10 Pro
- Python 3.9.1
- 내장 그래픽카드 : Intel UHD Graphics 630
```

### Sample Code
- `build.py`
```python
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, util, InputExample, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import math
import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#model_name = "kpfbert-base"
model_name = "D:\\pypro\\kpfbert\\"
train_batch_size = 16



word_embedding_model = models.Transformer(model_name)


# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)


model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

logging.info("Read AllNLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []

with open('KorNLUDatasets/KorNLI/snli_1.0_train.ko.tsv', "rt", encoding="utf-8") as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, label = line.split('\t')
        label = label2int[label.strip()]
        train_samples.append(InputExample(texts=[s1, s2], label=label))

train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))


#Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []

with open('KorNLUDatasets/KorSTS/tune_dev.tsv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, score = line.split('\t')
        score = score.strip()
        score = float(score) / 5.0
        dev_samples.append(InputExample(texts= [s1,s2], label=score))


dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

num_epochs = 1

warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
#           output_path=model_save_path
          )

model.save('build_results/kpfSBERT_nli')

train_batch_size = 16
num_epochs = 4
model_save_path = 'build_results/kpfSBERT'

# model = SentenceTransformer(model_name)
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
with open('KorNLUDatasets/KorSTS/tune_dev.tsv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, score = line.split('\t')
        score = score.strip()
        score = float(score) / 5.0
        dev_samples.append(InputExample(texts= [s1,s2], label=score))

with open('KorNLUDatasets/KorSTS/tune_test.tsv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, score = line.split('\t')
        score = score.strip()
        score = float(score) / 5.0
        test_samples.append(InputExample(texts= [s1,s2], label=score))

with open('KorNLUDatasets/KorSTS/tune_train.tsv', 'rt', encoding='utf-8') as fIn:
    lines = fIn.readlines()
    for line in lines:
        s1, s2, score = line.split('\t')
        score = score.strip()
        score = float(score) / 5.0
        train_samples.append(InputExample(texts= [s1,s2], label=score))

train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


# Development set: Measure correlation between cosine score and gold labels
logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))




# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
```

- `test.py`
```python
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, util, InputExample, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import math
import numpy as np

model_save_path = "build_results/kpfSBERT"

# TEST1 : sentesce similarity sorting with cosine similarity

model_path = model_save_path
model = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 속으로 밀었다.',
          '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
          '원숭이 한 마리가 드럼을 연주한다.',
          '치타 한 마리가 먹이 뒤에서 달리고 있다.']

corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['한 남자가 파스타를 먹는다.',
           '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
           '치타가 들판을 가로 질러 먹이를 쫓는다.']

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5
for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx in top_results[0:top_k]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))


# TEST2 : Clustering with k-means

model_path = model_save_path
model = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 속으로 밀었다.',
          '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
          '원숭이 한 마리가 드럼을 연주한다.',
          '치타 한 마리가 먹이 뒤에서 달리고 있다.',
          '한 남자가 파스타를 먹는다.',
          '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
          '치타가 들판을 가로 질러 먹이를 쫓는다.']

corpus_embeddings = model.encode(corpus)

# Then, we perform k-means clustering using sklearn:
from sklearn.cluster import KMeans

num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")
```

### Result
- 문장 유사성 검사에 대한 테스트 결과

```bash
PS D:\pypro\tbert\sbert> python test.py
======================


Query: 한 남자가 파스타를 먹는다.

Top 5 most similar sentences in corpus:
한 남자가 빵 한 조각을 먹는다. (Score: 0.5353)
한 남자가 음식을 먹는다. (Score: 0.5074)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.1085)
한 남자가 말을 탄다. (Score: 0.0791)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.0500)


======================


Query: 고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.

Top 5 most similar sentences in corpus:
원숭이 한 마리가 드럼을 연주한다. (Score: 0.5884)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.2077)
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.1205)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.1114)
한 남자가 말을 탄다. (Score: 0.0872)


======================


Query: 치타가 들판을 가로 질러 먹이를 쫓는다.

Top 5 most similar sentences in corpus:
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.8189)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.2705)
원숭이 한 마리가 드럼을 연주한다. (Score: 0.2093)
한 여자가 바이올린을 연주한다. (Score: 0.0981)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.0727)
Cluster  1
['그 여자가 아이를 돌본다.', '한 여자가 바이올린을 연주한다.']

Cluster  2
['한 남자가 음식을 먹는다.', '한 남자가 빵 한 조각을 먹는다.', '한 남자가 파스타를 먹는다.']

Cluster  3
['한 남자가 말을 탄다.', '두 남자가 수레를 숲 속으로 밀었다.', '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.']

Cluster  4
['치타 한 마리가 먹이 뒤에서 달리고 있다.', '치타가 들판을 가로 질러 먹이를 쫓는다.']

Cluster  5
['원숭이 한 마리가 드럼을 연주한다.', '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.']
```