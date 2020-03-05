from model_embeddings import ModelEmbeddings


class BaseDataAugmenter:
    def __init__(self, embed_size):
        self.model_embeddings = ModelEmbeddings(embed_size=embed_size)

    def augment(self, raw_train_data):
        '''
        @param raw_train_data: (List[List[words]], List[sentiment])
        
        @return: List[(embedded_vectors, sentiment)], with possibly extended BATCH_SIZE
        '''

        sentences, sentiments = raw_train_data
        sentences_embedded = self.model_embeddings.embed_sentence(sentences)

        return list(zip(sentences_embedded, sentiments))



class GaussianNoiseDataAugmenter:
    pass