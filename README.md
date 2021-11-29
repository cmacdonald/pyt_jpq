# PyTerrier JPQ

This is the PyTerrier plugin for Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance, known as [JPQ](https://github.com/jingtaozhan/JPQ/)

## Usage

JPQ has three stages: indexing, training and retrieval.

 - In indexing, documents are tokenised and encoded into a FAISS index. PyT_JPQ implements
 the IterDictIndexer-like API

```python
indexer = JPQIndexer(
    CHECKPOINT, 
    "./index",
    num_docs = num_docs)
indexer.index(pt.get_dataset("vaswani").get_corpus_iter())
```

 - During retrieval, a FAISS index can be used to retrieve documents.

```python
ret = JPQRetrieve("./index/", "OPQ96,IVF1,PQ96x8.index", CHECKPOINT)
res = ret.search("chemical")
```
 - During training, that FAISS index is rewritten.

```python
ret.fit(train_topics, train_qrels)
```

## Known Issues

 - easier access, i.e. automatic downloading of existing model checkpoints
 - better use of folders in the index_path


## Citations
 - [Zhan2021] Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance. Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, Shaoping Ma. In Proceedings of CIKM 2021. https://arxiv.org/abs/2108.00644

 - [Macdonald20]: Craig Macdonald, Nicola Tonellotto. Declarative Experimentation in Information Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271

## Credits

 - Buruo (Rachel) Guo, University of Glasgow
 - Craig Macdonald, University of Glasgow

Where possible, code in JPQ is directly called. Otherwise, some code was adapted from the 
original JPQ implementation by Jingtao Zhan et al.