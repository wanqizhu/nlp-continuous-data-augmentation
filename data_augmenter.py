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

        size = sentences[0].shape[1]
        _id = np.identity(size)

        sentences = sentences + [np.matmul(s, _id + np.random.normal(scale=self.std, size=_id.shape))
                                                    for s in (sentences*self.niters)]
        sentiments = sentiments + [s for s in (sentiments * self.niters)]

        return list(zip(sentences, sentiments))



class EmbedDimensionSwapDataAugmenter(BaseDataAugmenter):
    def __init__(self, nswaps=1, niters=3):
        super().__init__()
        self.nswaps = int(nswaps)
        self.niters = niters

    def augment(self, embedded_training_data):
        sentences, sentiments = embedded_training_data

        def swap(s):
            for _ in range(self.nswaps):
                dim1 = np.random.randint(s.shape[1])
                dim2 = np.random.randint(s.shape[1])
                s[:, dim1], s[:, dim2] = s[:, dim2], s[:, dim1].copy()
            return s

        sentences = sentences + [swap(s) for s in (sentences*self.niters)]
        sentiments = sentiments + [s for s in (sentiments * self.niters)]

        return list(zip(sentences, sentiments))
