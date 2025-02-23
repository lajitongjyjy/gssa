class Config(object):
    """
    Store hyper-parameters of neural networks。
    """
   
    num_classes = {'yahoo_answers': 10, 'ag_news': 4, 'dbpedia':14}
    word_max_len = {'yahoo_answers': 500, 'ag_news': 250, 'dbpedia':250}
    num_words = {'yahoo_answers': 50000,'ag_news': 50000, 'dbpedia':50000}
    embedding_size = {'yahoo_answers': 300, 'ag_news': 300, 'dbpedia':300}

    stop_words = ['the', 'a', 'an', 'to', 'of', 'and', 'with', 'as', 'at', 'by', 'is', 'was', 'are', 'were', 'be', 'he', 'she', 'they', 'their', 'this', 'that']
    word_index = {
        'the': 1,
        'a': 2,
        'an': 3,
        'to': 4,
        'of': 5,
        'and': 6,
        'with': 7,
        'as': 8,
        'at': 9,
        'by': 10,
        'is': 11,
        'was': 12,
        'are': 13,
        'were': 14,
        'be': 15,
        'he': 16,
        'she': 17,
        'they': 18,
        'their': 19,
        'this': 20,
        'that': 21,
    }


    
config = Config()
