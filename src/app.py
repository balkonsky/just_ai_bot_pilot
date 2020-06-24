import pandas as pd
import requests
import logging as log
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 8)

hostname = ''


def prepare_data_frame(excel_filename):
    df = pd.read_excel(excel_filename, sheet_name='Sheet1')
    for i, row in df.iterrows():
        if not pd.isna(df.at[i, 'utterance']):
            df.at[i, 'utterance'] = '"' + df['utterance'].values[i] + '"'
    return df


def save_df_to_csv(df, csv_train_filename):
    df.to_csv(csv_train_filename, encoding='utf-8', index=False)


def get_prepare_csv_file(csv_train_filename):
    return open(csv_train_filename, 'rb')


def count_accuracy(test_result_df):
    acc = accuracy_score(test_result_df['first_predict_intent'].tolist(), test_result_df['true_intent'].tolist())
    log.debug('count accuracy: {}'.format(acc))
    return acc


def print_confusion_matrix(true, predict):
    df_confusion = pd.crosstab(pd.Series(true, name='True'),
                               pd.Series(predict, name='Predict'))
    plt.matshow(df_confusion, cmap=plt.cm.gray_r)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()


def train_model(csv_train_file):
    log.info('train model...')
    resp = requests.post('{}/fit/'.format(hostname),
                         files={'file': csv_train_file}, timeout=10)
    if resp.ok:
        resp_json = resp.json()
        log.debug(
            'success train model with id: {}, number of samples: {}, intents : {}'.format(resp_json.get('model_id'),
                                                                                          resp_json.get(
                                                                                              'number_of_samples'),
                                                                                          resp_json.get('intents')))
        return resp_json.get('model_id')
    raise RuntimeError('Not success train model')


def test_model(csv_test_file, model_id):
    log.info('test model...')
    resp = requests.post('{}/predict/{}'.format(hostname, model_id),
                         files={'file': csv_test_file}, timeout=10)
    if resp.ok:
        test_result_df = pd.DataFrame(
            columns=['utterance', 'true_intent', 'first_predict_intent', 'first_predict_score', 'second_predict_intent',
                     'second_predict_score', 'third_predict_intent', 'third_predict_score'])
        df_test = pd.read_csv('cv_test.csv')
        resp_json = resp.json()
        for index, predict in enumerate(resp_json.get('predictions')):
            utterance = str(df_test.at[index, 'utterance'])
            true_intent = str(df_test.at[index, 'intent'])
            test_result_df = test_result_df.append(
                {'utterance': utterance, 'true_intent': true_intent,
                 'first_predict_intent': predict[0][0], 'first_predict_score': predict[0][1],
                 'second_predict_intent': predict[1][0], 'second_predict_score': predict[1][1],
                 'third_predict_intent': predict[2][0], 'third_predict_score': predict[2][1]}, ignore_index=True)
        return test_result_df
    raise RuntimeError('Not success test model')


def cross_validation(test_sample_split, df, test_iteration):
    log.info('cross validation process...')
    if test_iteration < 1:
        raise RuntimeError('iteration for test must be more or equals 1')
    accuracys = []
    for iter in range(test_iteration):
        df = shuffle(df)
        kf = KFold(n_splits=test_sample_split, shuffle=True, random_state=2)
        result = next(kf.split(df), None)
        training_df = df.iloc[result[0]]
        test_df = df.iloc[result[1]]
        try:
            save_df_to_csv(training_df, 'cv_training.csv')
            save_df_to_csv(test_df, 'cv_test.csv')
            model_id = train_model(get_prepare_csv_file('cv_training.csv'))
            res_test_df = test_model(get_prepare_csv_file('cv_test.csv'), model_id)
            save_df_to_csv(res_test_df, 'result_test_inter_{}.csv'.format(iter))
            accuracys.append(count_accuracy(res_test_df))
        except RuntimeError as e:
            log.error('except error with cross validation: {}'.format(e))
    log.info('mean accuracy: {}'.format(np.mean(accuracys)))


def main():
    prepare_df = prepare_data_frame('intents.xlsx')
    cross_validation(5, prepare_df, 10)


if __name__ == '__main__':
    log.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=log.DEBUG)
    main()
