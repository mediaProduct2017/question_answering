"""
@Project   : DuReader
@Module    : prepare_paragraphs.py
@Created   : 8/20/18 3:41 PM
@Desc      : 
"""
import pickle
import pprint
import time

from work4.elasticsearch2.extract_person import search_data_match
from work4.logger_setup import define_logger

logger = define_logger('work3.prepare_paragraphs')

if __name__ == '__main__':

    asked_sentences = search_data_match()
    logger.info('The context sentences for the questions:')
    time.sleep(0.5)
    pprint.pprint(asked_sentences)

    with open('paragraphs_messi.pkl', 'wb') as handle:
        pickle.dump(asked_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
