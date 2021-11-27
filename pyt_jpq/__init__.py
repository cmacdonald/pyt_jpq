import torch
import faiss

import pyterrier as pt
import pandas as pd
assert pt.started()
from typing import Union
from pyterrier.datasets import Dataset

from pyterrier.transformer import TransformerBase


def _preprocess_topicsqrels(topics, qrels, docno2docid, pid2offset, max_query_length=32):
    import tempfile
    from jpq.preprocess import write_query_rel
    from collections import defaultdict
    from itertools import count
    import os

    train_preprocess_dir = tempfile.mkdtemp()
    args = ArgsObject()
    args.data_dir  = train_preprocess_dir
    args.threads = 1
    args.max_query_length = max_query_length
    args.out_data_dir = train_preprocess_dir
    args.data_type = 1

    
    qid2int = defaultdict(count().__next__)
    import os
    with pt.io.autoopen(os.path.join(args.data_dir, "train.topics"), 'wt') as new_topics_file:
        for t in topics.itertuples():
            qidint = qid2int[t.qid]
            new_topics_file.write("%d\t%s\n" % (qidint, t.query))
    skipped=0
    with pt.io.autoopen(os.path.join(args.data_dir, "train.qrels"), 'wt') as new_qrels_file:
        for d in qrels.itertuples():
            if d.docno in docno2docid:
                docid = docno2docid[d.docno]
                new_qrels_file.write("%d\t0\t%d\t%d\n" % (qid2int[d.qid], docid, d.label))
            else:
                skipped+=1
    if skipped > 0:
        from warnings import warn
        warn("Could not match %d qrels to index" % skipped)

    write_query_rel(
        args, 
        pid2offset, "qid2offset", 
        "train.topics", "train.qrels", # our filenames, these dont matter
        "train-query", "train-qrel.tsv") # run_train requires names like this
    return train_preprocess_dir


class JPQIndexer(TransformerBase):
    
    def __init__(self, checkpoint_path, index_path, text_attr="text", num_docs=1000, segment_size=500_000, gpu=True):
        import os
        
        self.index_path = index_path
        self.checkpoint_path = checkpoint_path    
        self.segment_size = segment_size
        self.text_attr = text_attr
        self.gpu = gpu
        self.num_docs = num_docs
        
        args = type('', (), {})()
        args.model_dir = checkpoint_path

        if self.gpu:
            args.device = torch.device("cuda")
            args.n_gpu = torch.cuda.device_count()
        else:
            args.device = torch.device("cpu")
            args.n_gpu = 1
        

        args.subvector_num = 96
        args.max_seq_length=512
        args.max_doc_length=512
        args.preprocess_dir=index_path
        args.max_doc_character=10000
        
        args.eval_batch_size=128
        args.output_dir=index_path
        os.makedirs(args.output_dir, exist_ok=True)
        args.doc_embed_path=os.path.join(args.output_dir, "doc_embed.memmap")
        self.args = args

        
    def index(self, generator):
        import torch
        import more_itertools
        import pyterrier as pt
        import pickle
        import os
        from jpq.model import RobertaDot
        from transformers import RobertaConfig



        def tokenize_to_file(args, iter_dict, output_dir, line_fn, max_length, begin_idx, end_idx):
            from jpq.star_tokenizer import RobertaTokenizer
            import numpy as np
            from pyterrier import tqdm
            tokenizer = RobertaTokenizer.from_pretrained(
                "roberta-base", do_lower_case = True, cache_dir=None)
            os.makedirs(output_dir, exist_ok=True)
            data_cnt = end_idx - begin_idx
            ids_array = np.memmap(
                os.path.join(output_dir, "ids.memmap"),
                shape=(data_cnt, ), mode='w+', dtype=np.int32)
            token_ids_array = np.memmap(
                os.path.join(output_dir, "token_ids.memmap"),
                shape=(data_cnt, max_length), mode='w+', dtype=np.int32)
            token_length_array = np.memmap(
                os.path.join(output_dir, "lengths.memmap"),
                shape=(data_cnt, ), mode='w+', dtype=np.int32)
            pbar = tqdm(total=end_idx-begin_idx, desc=f"Tokenizing")
            for idx, psg in enumerate(iter_dict):
                psg['docid'] = idx
                qid_or_pid, token_ids, length = line_fn(args, psg, tokenizer)
                write_idx = idx - begin_idx
                ids_array[write_idx] = qid_or_pid
                token_ids_array[write_idx, :] = token_ids
                token_length_array[write_idx] = length
                pbar.update(1)
            pbar.close()

            assert write_idx == data_cnt - 1

        def PassagePreprocessingFn(args, passage : dict, tokenizer):
            from jpq.preprocess import pad_input_ids
            p_id = passage['docno']
            p_text = passage['text'].rstrip()

            # keep only first 10000 characters, should be sufficient for any
            # experiment that uses less than 500 - 1k tokens
            full_text = p_text[:args.max_doc_character]
            passage = tokenizer.encode(
                full_text,
                add_special_tokens=True,
                max_length=args.max_seq_length,
                truncation=True
            )
            passage_len = min(len(passage), args.max_seq_length)
            input_id_b = pad_input_ids(passage, args.max_seq_length)
            return p_id, input_id_b, passage_len

        args = self.args
        
        config = RobertaConfig.from_pretrained(args.model_dir, gradient_checkpointing=False)
        model = RobertaDot.from_pretrained(args.model_dir, config=config)
        model = model.to(args.device)
        embed_size = 768

        docno2id = {}
        def new_gen():
            for i, line in enumerate(generator):
                docno2id[line['docno']] = i
                yield line
        
        #todo chunking
        tokenize_to_file(args, new_gen(), os.path.join(self.index_path, "memmap"), PassagePreprocessingFn, 512, 0, self.num_docs)

        out_passage_path = self.index_path
        all_linecnt = self.num_docs

        import numpy as np
        token_ids_array = np.memmap(
            os.path.join(out_passage_path, "passages.memmap"),
            shape=(all_linecnt, args.max_seq_length), mode='w+', dtype=np.int32)
        idx = 0
        pid2offset = {}
        out_line_count = 0
        token_length_array = []
        splits_dir_lst = [os.path.join(self.index_path, "memmap")]
        for split_dir in splits_dir_lst:
            ids_array = np.memmap(
                os.path.join(split_dir, "ids.memmap"), mode='r', dtype=np.int32)
            split_token_ids_array = np.memmap(
                os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32)
            split_token_ids_array = split_token_ids_array.reshape(len(ids_array), -1)
            split_token_length_array = np.memmap(
                os.path.join(split_dir, "lengths.memmap"), mode='r', dtype=np.int32)
            for p_id, token_ids, length in zip(ids_array, split_token_ids_array, split_token_length_array):
                token_ids_array[idx, :] = token_ids
                token_length_array.append(length) 
                pid2offset[p_id] = idx
                idx += 1
                if idx < 3:
                    print(str(idx) + " " + str(p_id))
                out_line_count += 1
        assert len(token_length_array) == len(token_ids_array) == idx
        np.save(os.path.join(out_passage_path, "passages_length.npy"), np.array(token_length_array))
        token_ids_array = None

        args.out_data_dir = self.index_path
        pid2offset_path = os.path.join(
            args.out_data_dir,
            "pid2offset.pickle",
        )
        docno2docid_path = os.path.join(
            args.out_data_dir,
            "docno2docid.pickle",
        )


        with open(pid2offset_path, 'wb') as handle:
            pickle.dump(pid2offset, handle, protocol=4)
        
        with open(docno2docid_path, 'wb') as handle:
            pickle.dump(docno2id, handle, protocol=4)
        

        print("done saving pid2offset")

        with open(os.path.join(self.index_path, "passages_meta"), 'wt') as metafile:
            import json
            metafile.write(json.dumps({'total_number': self.num_docs, 'embedding_size': 512, 'type': 'int32'}))

        from jpq.run_init import doc_inference
        doc_inference(model, args, embed_size)

        save_index_path = os.path.join(args.output_dir, f"OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x8.index")

        doc_embeddings = np.memmap(args.doc_embed_path, dtype=np.float32, mode="r")
        doc_embeddings = doc_embeddings.reshape(-1, embed_size)

        import faiss

        if self.gpu:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024*1024*512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = args.subvector_num >= 56

        faiss.omp_set_num_threads(32)
        dim = embed_size
        index = faiss.index_factory(dim, 
            f"OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x8", faiss.METRIC_INNER_PRODUCT)
        index.verbose = True    
        if self.gpu:
           index = faiss.index_cpu_to_gpu(res, 0, index, co)
        
        index.train(doc_embeddings)
        index.add(doc_embeddings)
        if self.gpu:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, save_index_path)



        return self.index_path

class ArgsObject:
    pass

class JPQRetrieve(TransformerBase) :
    def __init__(self, index_path, faiss_name, model_path__queryencoder_dir, max_query_length=32, batch_size=1, topk=100, gpu=True):
        from jpq.star_tokenizer import RobertaTokenizer
        from jpq.model import RobertaDot
        from transformers import RobertaConfig
        from jpq.run_retrieval import load_index
        import os, pickle

        self.gpu = gpu

        # load the model 
        config_class, model_class = RobertaConfig, RobertaDot
        config = config_class.from_pretrained(model_path__queryencoder_dir)
        self.model = model_class.from_pretrained(model_path__queryencoder_dir, config=config,)
        self.topk = topk
        #load the tokeniser
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", do_lower_case = True, cache_dir=None)

        self.faiss_path = os.path.join(index_path, faiss_name)
        #load the index
        self.index = load_index(self.faiss_path, use_cuda=gpu, faiss_gpu_index=0)
        self.max_query_length = max_query_length

        pid2offset_path = os.path.join(
            index_path,
            "pid2offset.pickle",
        )

        docno2docid_path = os.path.join(
            index_path,
            "docno2docid.pickle",
        )

        with open(pid2offset_path, 'rb') as handle:
            self.pid2offset = pickle.load(handle)
        with open(docno2docid_path, 'rb') as handle:
            self.docno2docid = pickle.load(handle)
        self.docid2docno = [None] * len(self.docno2docid)
        for docno, docid in self.docno2docid.items():
            self.docid2docno[docid] = docno


    def fit(self, train_topics, train_qrels):
        import faiss
        args = type('', (), {})()
        args.model_device = torch.device("cpu") #torch.cuda.device(0)

        opq_index = self.index
        vt = faiss.downcast_VectorTransform(opq_index.chain.at(0))            
        assert isinstance(vt, faiss.LinearTransform)
        opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
        opq_transform = torch.FloatTensor(opq_transform).to(args.model_device)

        ivf_index = faiss.downcast_index(opq_index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)
        pq_codes = torch.LongTensor(pq_codes).to(args.model_device)

        centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
        centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)
        coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
        centroid_embeds += coarse_embeds.reshape(ivf_index.pq.M, -1, ivf_index.pq.dsub)
        faiss.copy_array_to_vector(centroid_embeds.ravel(), ivf_index.pq.centroids)
        coarse_embeds[:] = 0
        faiss.copy_array_to_vector(coarse_embeds.ravel(), coarse_quantizer.xb)

        centroid_embeds = torch.FloatTensor(centroid_embeds).to(args.model_device)
        centroid_embeds.requires_grad = True

        train_preprocess_dir = _preprocess_topicsqrels(
            train_topics, train_qrels, 
            self.docno2docid,
            self.pid2offset, max_query_length=self.max_query_length)

        from jpq.run_train import train
        args = ArgsObject()
        import tempfile, os
        args.log_dir = tempfile.mkdtemp()
        args.model_save_dir = "./newmodel"
        args.n_gpu = 1
        args.model_device = torch.device(f"cuda:0")
        args.model_device = torch.device("cpu")
        os.makedirs(args.model_save_dir, exist_ok=True)
        args.init_index_path = self.faiss_path
        args.gpu_search = self.gpu
        args.preprocess_dir = train_preprocess_dir
        args.max_seq_length = 32
        args.centroid_lr = 1e-4
        args.loss_neg_topK = 200
        args.gradient_accumulation_steps = 1
        args.num_train_epochs = 6 #The number of training epochs is set to 6. In fact, the performance is already quite satisfying after 1 or 2 epochs. Each epoch costs less than 2 hours on our machine.
        args.warmup_steps = 2000
        args.logging_steps = 100
        args.weight_decay = 0.01
        args.lambda_cut = 10
        args.centroid_weight_decay = 0
        args.threads = 1
        args.lr = 5e-6
        args.seed = 42
        args.max_grad_norm = 1
        args.adam_epsilon = 1e-8
        args.train_batch_size = 32
        train(args, self.model, pq_codes, centroid_embeds, opq_transform, opq_index)

        # now reopen the new faiss index
        new_index = os.path.join(args.model_save_dir, f'epoch-{args.num_train_epochs}', 
                os.path.basename(args.init_index_path))
        from jpq.run_retrieval import load_index
        self.index = load_index(self.faiss_path, use_cuda=self.gpu, faiss_gpu_index=0)


    #allows a JPQ ranker to be obained from a dataset
    def from_dataset(dataset : Union[str,Dataset], 
            variant : str = None, 
            version='latest',            
            **kwargs):

        from pyterrier.batchretrieve import _from_dataset

        #JPQRetrieve doesnt match quite the expectations, so we can use a wrapper fn
        def _JPQRetrieveconstruct(folder, **kwargs):
            import os
            checkpoint_path = kwargs.get('checkpoint_path')
            del kwargs['checkpoint_path']
            return JPQRetrieve(checkpoint_path, folder, **kwargs)

        return _from_dataset(dataset, 
                                variant=variant, 
                                version=version, 
                                clz=_JPQRetrieveconstruct, **kwargs)

    def __str__(self):
        return "JPQ"

    def transform(self ,topics: pd.DataFrame) -> pd.DataFrame:  
        """
        input columns: qid, query
        output columns: qid, docid, docno, score, rank
        """
        def QueryPreprocessingFn(query, tokenizer, max_query_length):   
            passage = tokenizer.encode(
                query.rstrip(),
                add_special_tokens=True,
                max_length=max_query_length,
                truncation=True)
            passage_len = min(len(passage), max_query_length)
            # input_id_b = pad_input_ids(passage, max_query_length)
            return passage, passage_len

        rtr = []
        for row in topics.itertuples(): 
            ids, length = QueryPreprocessingFn(row.query, self.tokenizer, max_query_length=self.max_query_length )

            # apply the preprocessing. - see https://github.com/jingtaozhan/JPQ/blob/main/preprocess.py#L355
            with torch.no_grad():
                #get the query_embeds
                a = torch.tensor([ids])
                b = torch.ones(1,len(ids))
                query_embeds = self.model(
                    input_ids=a, 
                    attention_mask=b, 
                    ).detach().cpu().numpy()
            #search the index
            scores, batch_results = self.index.search(query_embeds, self.topk)
            
            docid_list=batch_results.tolist()[0]
            df = pd.DataFrame({'qid':str(row.qid),'docid':docid_list})
            df['docno'] = df.docid.map(lambda docid : self.docid2docno[docid])
            df['score'] = scores[0]
            df['rank'] = [i for i , _ in enumerate(docid_list)]
            rtr.append(df)

        res = pd.concat(rtr)

        return res