import warnings
warnings.filterwarnings('ignore')
from gensim.models import FastText
from cntk.layers.blocks import _INFERRED
from collections import defaultdict
from config import *
import numpy as np
import cntk as C

"""
CUDNN approach rnn stack
"""
def OptimizedRnnStack(hidden_dim, num_layers=1, recurrent_op='lstm', bidirectional=True, 
                      use_cudnn=True, name=''):
    if use_cudnn:
        W = C.parameter(_INFERRED + (hidden_dim,), init=C.glorot_uniform())
        def func(x):
            return C.optimized_rnnstack(x, W, hidden_dim, num_layers, bidirectional, 
                                        recurrent_op=recurrent_op, name=name)
        return func
    else:
        raise NotImplementedError

class Model:
    def __init__(self):
        self.embedding_path = embedding_path
        self.fmodel = FastText.load(self.embedding_path)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_dim = len(self.fmodel.wv.vocab) + 1
        self.vocab = defaultdict(int)
        self.num_emotions = 8
        self.dropout = dropout
        self.use_cudnn = True
        
        counter = 1
        for vocab in self.fmodel.wv.vocab:
            self.vocab[vocab] = counter
            counter += 1
        
    def embed(self):
        vec = np.zeros((self.word_dim, self.embedding_dim), dtype=np.float32)

        for v in self.fmodel.wv.vocab:
            vec[self.vocab[v]] = self.fmodel.wv[v]
        
        embedding = C.parameter(shape=vec.shape, init=vec)
        
        def func(context):
            return C.times(context, embedding)
        
        return func
        
    def input_layer(self, token):
        token_ph = C.placeholder()
        
        input_words = C.placeholder(shape=(self.word_dim))
        
        embedded = self.embed()(input_words)
        processed = OptimizedRnnStack(self.hidden_dim,
                       num_layers=1, bidirectional=True, use_cudnn=True, name='input_rnn')(embedded)
        
        processed_token = processed.clone(C.CloneMethod.share, {input_words: token_ph})
        
        return C.as_block(
            processed_token,
            [(token_ph, token)],
            'input_layer',
            'input_layer'
        )
    
    def attention_layer(self, c1, c2, layer):        
        q_processed = C.placeholder(shape=(2*self.hidden_dim,))
        p_processed = C.placeholder(shape=(2*self.hidden_dim,))

        qvw, qvw_mask = C.sequence.unpack(q_processed, padding_value=0).outputs

        wq = C.parameter(shape=(2*self.hidden_dim, 2*self.hidden_dim), init=C.glorot_uniform())
        wp = C.parameter(shape=(2*self.hidden_dim, 2*self.hidden_dim), init=C.glorot_uniform())
        wg = C.parameter(shape=(8*self.hidden_dim, 8*self.hidden_dim), init=C.glorot_uniform())
        v = C.parameter(shape=(2*self.hidden_dim, 1), init=C.glorot_uniform())

        # seq[tensor[2d]] p_len x 2d
        wpt = C.reshape(C.times(p_processed, wp), (-1, 2*self.hidden_dim))

        # q_len x 2d
        wqt = C.reshape(C.times(qvw, wq), (-1, 2*self.hidden_dim))
        
        # seq[tensor[q_len]]
        S = C.reshape(C.times(C.tanh(C.sequence.broadcast_as(wqt, p_processed) + wpt), v), (-1))

        qvw_mask_expanded = C.sequence.broadcast_as(qvw_mask, p_processed)

        # seq[tensor[q_len]]
        S = C.element_select(qvw_mask_expanded, S, C.constant(-1e+30))
        
        # seq[tensor[q_len]]
        A = C.softmax(S, axis=0)

        # seq[tensor[2d]]
        swap_qvw = C.swapaxes(qvw)
        cq = C.reshape(C.reduce_sum(A * C.sequence.broadcast_as(swap_qvw, A), axis=1), (-1))

        # seq[tensor[4d]]
        uc_concat = C.splice(p_processed, cq, p_processed * cq, cq * cq)
        
        # seq[tensor[4d]]
        gt = C.tanh(C.times(uc_concat, wg))
        
        # seq[tensor[4d]]
        uc_concat_star = gt * uc_concat
 
        # seq[tensor[4d]]
        vp = C.layers.Sequential([
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, num_layers=2,
                use_cudnn=self.use_cudnn, name=layer+'_attention_rnn')])(uc_concat_star)
        
        return C.as_block(
            vp,
            [(p_processed, c1), (q_processed, c2)],
            'attention_layer_' + layer,
            'attention_layer_' + layer)
        
    def model(self):
        token_axis = C.Axis.new_unique_dynamic_axis('token_axis')
        b = C.Axis.default_batch_axis()
        
        token = C.input_variable(self.word_dim, dynamic_axes=[b, token_axis], name='token')
        
        # 8 classes
        emotion = C.input_variable(self.num_emotions, dynamic_axes=[b], name='emotion')
        
        processed_token = self.input_layer(token)
        att = self.attention_layer(processed_token, processed_token, 'attention')
        
        test = C.sequence.last(att)
        test = C.layers.Stabilizer()(test)
        
        test_w = C.parameter(shape=(2*self.hidden_dim, self.num_emotions), init=C.glorot_uniform())
        test_v = C.parameter(shape=(self.num_emotions), init=C.glorot_uniform())
        
        out = C.softmax(C.times(test, test_w) + test_v)
        loss = C.binary_cross_entropy(out, emotion)
        
        f1 = C.losses.fmeasure(C.hardmax(out), emotion)
        
        return out, loss, f1
