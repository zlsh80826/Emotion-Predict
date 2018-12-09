import warnings
warnings.filterwarnings('ignore')
from config import *
from helper import *
from model import Model
import cntk as C
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd

def inference(model, data):
    p = Model()
    model = C.load_model(model)

    prob = model.outputs[0]
    loss = C.as_composite(model.outputs[1].owner)

    mb_test, map_test = deserialize(loss, data, p, 
                                    randomize=False, repeat=False, is_test=True)
    token = argument_by_name(loss, 'token')
    
    results = []
    total_samples = 411972
        
    with tqdm(total=total_samples, ncols=79) as progress_bar:
        while True:
            data = mb_test.next_minibatch(4, input_map=map_test)
            if not data:
                break
            out = model.eval(data, outputs=[prob])
            results.extend(out)
            progress_bar.update(len(data))
    assert(len(results) == total_samples)
    return results    

def write_answer(datadir, results, output):
    answer = np.argmax(results, axis=1)
    test_df = pd.read_pickle('{}/test.pkl'.format(datadir))
    tid = test_df['tweet_id']

    emotion_dict = {0: 'anticipation', 1: 'sadness', 2: 'fear', 3: 'joy', 
                    4: 'anger', 5: 'trust', 6: 'disgust', 7:'surprise'}

    answer_pd = pd.DataFrame(columns=['id', 'emotion'])
    answer_pd['id'] = tid
    emotion = list()
    for a in answer:
        emotion.append(emotion_dict[a])
    answer_pd['emotion'] = emotion
    answer_pd.to_csv(output, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference model')
    parser.add_argument('--model', help='Model weights file', required=False, default='../model/' + version + '/0')
    parser.add_argument('--test', help='Test file in ctf format', required=False, default='../data/test.ctf')
    parser.add_argument('--answer', help='Answer file name', required=False, default=version + '_submission.csv')
    parser.add_argument('--datadir', help='Data directory', required=False, default='../data')
    args = parser.parse_args()

    results = inference(args.model, args.test)
    write_answer(args.datadir, results, args.answer)
