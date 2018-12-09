import ujson as json
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    idf = pd.read_csv('../data/data_identification.csv', header=0)

    id_ = dict()
    for i in tqdm(range(len(idf))):
        id_[idf.iloc[i]['tweet_id']] = idf.iloc[i]['identification']

    data = pd.DataFrame(columns=['tweet_id', 'text', 'hashtags', 'score', 
                                 'index', 'crawldata', 'type', 'dataset'])
    crawldate = list()
    tweet_id = list()
    hashtags = list()
    dataset = list()
    score = list()
    index = list()
    type_ = list()
    text = list()

    with open('../data/tweets_DM.json', 'r') as file:
        for line in tqdm(file):
            js = json.loads(line)
            tweet_id.append(js['_source']['tweet']['tweet_id'])
            text.append(js['_source']['tweet']['text'])
            hashtags.append(js['_source']['tweet']['hashtags'])
            score.append(js['_score'])
            index.append(js['_index'])
            crawldate.append(js['_crawldate'])
            type_.append(js['_type'])
            dataset.append(id_[js['_source']['tweet']['tweet_id']])

    data = pd.DataFrame({
        'tweet_id': tweet_id, 
        'text': text, 
        'hashtags': hashtags,
        'score': score, 
        'index': index, 
        'crawldata': crawldate, 
        'type': type_, 
        'dataset': dataset
    })

    train_df = data[data['dataset'] == 'train']
    test_df = data[data['dataset'] == 'test']

    print('Number of training data:', len(train_df))
    print('Number of testing data:', len(test_df))

    emotions = pd.read_csv('../data/emotion.csv', header=0)
    emotion_dict = dict()

    for i in tqdm(range(len(emotions))):
        emotion_dict[emotions.iloc[i]['tweet_id']] = emotions.iloc[i]['emotion']

    emotion_list = list()

    for i in tqdm(range(len(train_df))):
        emotion_list.append(emotion_dict[train_df.iloc[i]['tweet_id']])

    train_df['emotion'] = emotion_list

    train_df.to_pickle('../data/train.pkl')
    test_df.to_pickle('../data/test.pkl')
