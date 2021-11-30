import torch
import faiss

import pyterrier as pt
import pandas as pd
assert pt.started()
from typing import Union
from warnings import warn
from pyterrier.datasets import Dataset

from pyterrier.transformer import TransformerBase

DEFAULT_TRAINING_ARGS = {
    'n_gpu' : 1,
    'max_seq_length' : 32,
    'centroid_lr' : 1e-4,
    'loss_neg_topK' : 200,
    'gradient_accumulation_steps' : 1,
    'num_train_epochs' : 6, # "The number of training epochs is set to 6. In fact, the performance is already quite satisfying after 1 or 2 epochs. Each epoch costs less than 2 hours on our machine."
    'warmup_steps' : 2000,
    'logging_steps' : 100,
    'weight_decay' : 0.01,
    'lambda_cut' : 10,
    'centroid_weight_decay' : 0,
    'threads' : 1,
    'lr' : 5e-6,
    'seed' : 42,
    'max_grad_norm' : 1,
    'adam_epsilon' : 1e-8,
    'train_batch_size' : 32
}

def _preprocess_topicsqrels(topics, qrels, docno2docid, pid2offset, max_query_length=32):
    import tempfile, os
    from collections import defaultdict
    from itertools import count
    import pyterrier as pt

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

    import jpq.preprocess
    jpq.preprocess.tqdm = pt.tqdm
    jpq.preprocess.write_query_rel(
        args, 
        pid2offset, "qid2offset", 
        "train.topics", "train.qrels", # our filenames, these dont matter
        "train-query", "train-qrel.tsv") # run_train requires names like this
    return train_preprocess_dir


class JPQIndexer(TransformerBase):
    
    def __init__(self, checkpoint_path, index_path, text_attr="text", segment_size=500_000, gpu=True):
        import os
        
        self.index_path = index_path
        self.checkpoint_path = checkpoint_path    
        self.segment_size = segment_size
        self.text_attr = text_attr
        self.gpu = gpu
        
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


        def tokenize_to_file(args, tokenizer, iter_dict, output_dir, line_fn, max_length, max_docs):
            
            import numpy as np
            from pyterrier import tqdm
            os.makedirs(output_dir, exist_ok=True)
           
            ids_array = np.memmap(
                os.path.join(output_dir, "ids.memmap"),
                shape=(max_docs, ), mode='w+', dtype=np.int32)
            token_ids_array = np.memmap(
                os.path.join(output_dir, "token_ids.memmap"),
                shape=(max_docs, max_length), mode='w+', dtype=np.int32)
            token_length_array = np.memmap(
                os.path.join(output_dir, "lengths.memmap"),
                shape=(max_docs, ), mode='w+', dtype=np.int32)
            for idx, psg in tqdm(enumerate(iter_dict), total=max_docs, desc=f"Tokenizing"):
                psg['docid'] = idx
                qid_or_pid, token_ids, length = line_fn(args, psg, tokenizer)
                ids_array[idx] = qid_or_pid
                token_ids_array[idx, :] = token_ids
                token_length_array[idx] = length
            return idx

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
        
        from jpq.star_tokenizer import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(
                "roberta-base", do_lower_case = True, cache_dir=None)

        #todo chunking
        from more_itertools import chunked
        splits_dir_lst=[]
        splits_doc_count=[]
        for chunk_id, subset in enumerate(chunked(new_gen(), self.segment_size)):
            target_dir = os.path.join(self.index_path, "memmap_%d" % chunk_id)
            segment_size = tokenize_to_file(args, tokenizer, subset, target_dir, PassagePreprocessingFn, 512, self.segment_size)
            splits_dir_lst.append(target_dir)
            splits_doc_count.append(segment_size)
        
        self.num_docs = sum(splits_doc_count)
        out_passage_path = self.index_path
        
        import numpy as np
        token_ids_array = np.memmap(
            os.path.join(out_passage_path, "passages.memmap"),
            shape=(self.num_docs, args.max_seq_length), mode='w+', dtype=np.int32)
        idx = 0
        pid2offset = {}
        out_line_count = 0
        token_length_array = []
        from pyterrier import tqdm
        for split_dir, split_size in tqdm(zip(splits_dir_lst, splits_doc_count), desc="Merging segments", total=len(splits_doc_count)):
            ids_array = np.memmap(
                os.path.join(split_dir, "ids.memmap"), mode='r', dtype=np.int32)
            split_token_ids_array = np.memmap(
                os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32)
            split_token_ids_array = split_token_ids_array.reshape(len(ids_array), -1)
            split_token_length_array = np.memmap(
                os.path.join(split_dir, "lengths.memmap"), mode='r', dtype=np.int32)
            for local_pid in range(split_size): #TODO could we do this as a batch copy?
                #for p_id, token_ids, length in zip(ids_array, split_token_ids_array, split_token_length_array):
                p_id = ids_array[local_pid]
                token_ids = split_token_ids_array[local_pid]
                length = split_token_length_array[local_pid]
                
                token_ids_array[idx, :] = token_ids
                token_length_array.append(length) 
                #pid2offset[p_id] = idx
                idx += 1
                out_line_count += 1
        
        pid2offset = {x : x for x in range(idx)}
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

        # passages_meta is required by TextTokenIdsCache
        with open(os.path.join(self.index_path, "passages_meta"), 'wt') as metafile:
            import json
            metafile.write(json.dumps({
                'total_number': self.num_docs, 
                'embedding_size': 512, 
                'type': 'int32',
            }))
        

        from jpq.run_init import doc_inference
        doc_inference(model, args, embed_size)

        save_index_path = f"OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x8.index" 

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
        print("Index is %s" % str(index))
        faiss.write_index(index, os.path.join(args.output_dir, save_index_path))

        with open(os.path.join(self.index_path, "passages_meta"), 'wt') as metafile:
            import json
            metafile.write(json.dumps({
                'total_number': self.num_docs, 
                'embedding_size': 512, 
                'type': 'int32',
                'current_faiss_index' : save_index_path
            }))

        return self.index_path

class ArgsObject:
    pass

class JPQRetrieve(TransformerBase) :
    def __init__(self, index_path, model_path__queryencoder_dir, max_query_length=32, batch_size=1, topk=100, gpu=True, gpu_search=True, faiss_name=None):
        from jpq.star_tokenizer import RobertaTokenizer
        from jpq.model import RobertaDot
        from transformers import RobertaConfig
        from jpq.run_retrieval import load_index
        import os, pickle, json

        self.index_path = index_path
        self.gpu = gpu
        self.gpu_search = gpu_search

        # load the model 
        config_class, model_class = RobertaConfig, RobertaDot
        config = config_class.from_pretrained(model_path__queryencoder_dir)
        self.model = model_class.from_pretrained(model_path__queryencoder_dir, config=config,)
        if self.gpu:
            self.model = self.model.to(torch.device("cuda:0"))
        self.topk = topk
        #load the tokeniser
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", do_lower_case = True, cache_dir=None)

        meta_file = os.path.join(self.index_path, "passages_meta")
        if os.path.exists(meta_file)
            with open(meta_file, 'rt') as metafile:
                self.meta = json.load(metafile)
        else:
            self.meta={}
            warn("No passages_meta present, params assumed from constructor args")

        if faiss_name is None:
            faiss_name = self.meta['current_faiss_index']

        self.faiss_path = os.path.join(self.index_path, faiss_name)
        # load the FAISS index
        self.index = load_index(self.faiss_path, use_cuda=gpu_search, faiss_gpu_index=0)
        self.max_query_length = max_query_length

        pid2offset_path = os.path.join(
            index_path,
            "pid2offset.pickle",
        )

        docno2docid_path = os.path.join(
            index_path,
            "docno2docid.pickle",
        )

        if os.path.exists(pid2offset_path):
            with open(pid2offset_path, 'rb') as handle:
                self.pid2offset = pickle.load(handle)
        else:
            warn("No pid2offset present, training not possible")

        if os.path.exists(docno2docid_path):
            with open(docno2docid_path, 'rb') as handle:
                self.docno2docid = pickle.load(handle)
        else:
            num_docs = self.index.ntotal
            self.docno2docid = {str(d) : d for d in range(num_docs)}
            warn("No docno2docid present, assuming str(docid) == docno")
        
        self.docid2docno = [None] * len(self.docno2docid)
        for docno, docid in self.docno2docid.items():
            self.docid2docno[docid] = docno


    def fit(self, train_topics, train_qrels, **fit_params):

        # get the PQ vectors that we are training
        import faiss
        args = type('', (), {})()
        args.model_device = torch.device("cuda:0") if self.gpu else torch.device("cpu")

        opq_index = self.index
        # starting code assumes its a CPU index
        if "GpuIndex" in repr(faiss.downcast_index(opq_index.index)):
            self.index = opq_index = faiss.index_gpu_to_cpu(opq_index)
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

        # pre-tokenize the training topics
        train_preprocess_dir = _preprocess_topicsqrels(
            train_topics, train_qrels, 
            self.docno2docid,
            self.pid2offset, max_query_length=self.max_query_length)
        

        # now perform the training
        import tempfile, os
        args = ArgsObject()
        args.log_dir = tempfile.mkdtemp()
        print("JPQ training is logging to %s" % args.log_dir)
        args.model_save_dir = "./newmodel" #TODO fix
        args.model_device = torch.device(f"cuda:0") if self.gpu else torch.device("cpu")
        os.makedirs(args.model_save_dir, exist_ok=True)
        args.init_index_path = self.faiss_path
        args.gpu_search = self.gpu_search
        args.preprocess_dir = train_preprocess_dir
        train_params = DEFAULT_TRAINING_ARGS.copy()
        train_params.update(fit_params)
        for k,v in train_params.items():
            setattr(args, k, v)

        import jpq.run_train
        jpq.run_train.tqdm = pt.tqdm
        jpq.run_train.train(args, self.model, pq_codes, centroid_embeds, opq_transform, opq_index)

        # now reopen the new faiss index
        new_index = os.path.join(args.model_save_dir, f'epoch-{args.num_train_epochs}', 
                os.path.basename(args.init_index_path))
        from jpq.run_retrieval import load_index
        self.index = load_index(self.faiss_path, use_cuda=self.gpu_search, faiss_gpu_index=0)

        #TODO: write new_index to correct folder
        #TODO: update passages_meta file
        

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

        model_device = torch.device("cuda:0") if self.gpu else torch.device("cpu")
        rtr = []
        for row in topics.itertuples(): 
            ids, length = QueryPreprocessingFn(row.query, self.tokenizer, max_query_length=self.max_query_length )

            # apply the preprocessing. - see https://github.com/jingtaozhan/JPQ/blob/main/preprocess.py#L355
            with torch.no_grad():
                #get the query_embeds
                a = torch.tensor([ids]).to(model_device)
                b = torch.ones(1,len(ids)).to(model_device)
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
