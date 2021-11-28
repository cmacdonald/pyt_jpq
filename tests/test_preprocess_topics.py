import unittest
import tempfile
import pandas as pd
class TestPreprocess(unittest.TestCase):

    def test_write_topics(self):
        topics = pd.DataFrame([['q1', 'query A'], ['q2', 'query B']], columns=['qid', 'query'])
        qrels = pd.DataFrame([['q1', 'doc1', 1], ['q2', 'doc2', 2]], columns=['qid', 'docno', 'label'])

        import pyt_jpq
        import pyterrier as pt
        docno2docid = {'doc1' : 0, 'doc2' : 1}
        pid2offset = {0: 0, 1:10} 
        dir = pyt_jpq._preprocess_topicsqrels(topics, qrels, docno2docid, pid2offset, max_query_length=32)
        print('\n'.join(pt.io.find_files(dir)))

    
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