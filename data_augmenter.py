
def BaseDataAugmenter():
    def __init__(self):
        pass

    def augment(self, sentences_embedded, sentences_length, sentiments):
        '''
        @param sentences_embedded: Tensor of shape (max_sentence_length, batch_size, embed_size)
        @param sentences_length: List of ints of length (batch_size)
        @sentiments: List of labels (0-5) of length (batch_size)

        @return: the above three variables, with possibly extended BATCH_SIZE
        '''

        return sentences_embedded, sentences_length, sentiments