import warnings
warnings.filterwarnings('ignore')
from gensim.models import FastText
from tqdm import tqdm
import numpy as np
import cntk as C
from model import Model
from config import *
from helper import *

"""
Trainer
"""
def train():
    model = Model()
    z, loss, acc = model.model()
    
    progress_writers = [C.logging.ProgressPrinter(
                            num_epochs = max_epochs,
                            freq = log_freq,
                            tag = 'Training',
                            log_to_file = 'log/log_' + version)]
    
    lr = C.learning_parameter_schedule(learning_rate, minibatch_size=None, epoch_size=None)
    learner = C.adadelta(z.parameters, lr)
    trainer = C.Trainer(z, (loss, acc), learner, progress_writers)
    
    mb_source, input_map = deserialize(loss, train_data, model)
    mb_valid, valid_map = deserialize(loss, valid_data, model)
    
    try:
        trainer.restore_from_checkpoint('../model/' + version)
    except Exception:
        print('No checkpoint.')
    
    for epoch in range(max_epochs):
        # train
        num_seq = 0
        with tqdm(total=epoch_size, ncols=79) as progress_bar:
            while True:
                data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
                trainer.train_minibatch(data)
                num_seq += trainer.previous_minibatch_sample_count
                progress_bar.update(trainer.previous_minibatch_sample_count)
                if num_seq >= epoch_size:
                    break
            trainer.summarize_training_progress()
            trainer.save_checkpoint('../model/' + version + '/' + str(epoch))     
            
        # validation
        num_seq = 0    
        with tqdm(total=num_validation, ncols=79) as valid_progress_bar:
            while True:
                data = mb_valid.next_minibatch(minibatch_size, input_map=valid_map)
                if not data:
                    break
                trainer.test_minibatch(data)
                num_seq += len(data)
                valid_progress_bar.update(len(data))
                if num_seq >= num_validation:
                    break
            trainer.summarize_test_progress()

if __name__ == '__main__':
    train()
