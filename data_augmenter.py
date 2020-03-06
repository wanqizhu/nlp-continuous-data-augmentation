import numpy as np

class BaseDataAugmenter:
    def __init__(self):
        pass

    def augment(self, embedded_training_data):
        '''
        @param embedded_training_data: (List[np.ndarrays (unpadded embeddings)], List[sentiment])
        
        @return: List[(embedded_vectors, sentiment)], with possibly extended BATCH_SIZE
        '''

        sentences, sentiments = embedded_training_data
        return list(zip(sentences, sentiments))



class GaussianNoiseDataAugmenter(BaseDataAugmenter):
    def __init__(self, std=0.01, niters=3):
        super().__init__()
        self.std = std
        self.niters = niters


    def augment(self, embedded_training_data):
        sentences, sentiments = embedded_training_data

        sentences = sentences + [s + np.random.normal(scale=self.std, size=s.shape)
                                                    for s in (sentences*self.niters)]
        sentiments = sentiments + [s for s in (sentiments * self.niters)]

        return list(zip(sentences, sentiments))



class NoisyIdentityDataAugmenter(BaseDataAugmenter):
    def __init__(self, std=0.01, niters=3):
        super().__init__()
        self.std = std
        self.niters = niters


    def augment(self, embedded_training_data):
        sentences, sentiments = embedded_training_data

        s = s.shape[1]
        _id = np.identity(s)

        sentences = sentences + [np.matmul(s, _id + np.random.normal(scale=self.std, size=_id.shape))
                                                    for s in (sentences*self.niters)]
        sentiments = sentiments + [s for s in (sentiments * self.niters)]

        return list(zip(sentences, sentiments))


