import unittest
import pandas as pd
import tempfile
CHECKPOINT="./m96/"
class TestIndexing(unittest.TestCase):

    def test_index_1k(self):
        import pyterrier as pt
        from pyt_jpq import JPQIndexer, JPQRetrieve
        import os
        import shutil
        shutil.rmtree("./index", ignore_errors=True)
        num_docs = 1000
        indexer = JPQIndexer(
            CHECKPOINT, 
            "./index", #os.path.dirname(self.test_dir), 
            num_docs = num_docs,
            gpu=False)

        iter = pt.get_dataset("vaswani").get_corpus_iter()
        indexer.index([ next(iter) for i in range(num_docs) ])

        ret = JPQRetrieve("./index/OPQ96,IVF1,PQ96x8.index", CHECKPOINT, gpu=False)
        res = ret.search("chemical")
        self.assertTrue(len(res) > 0)

        ret.fit(
            pt.get_dataset("vaswani").get_topics(),
            pt.get_dataset("vaswani").get_qrels()
        )


    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass