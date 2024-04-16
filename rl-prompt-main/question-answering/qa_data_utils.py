from torch.utils.data import Dataset

class QuestionAnsweringDataset(Dataset):
    """
    TODO
    """

    def __init__(self, 
                 source_contexts,
                 source_questions, 
                 target_labels,
                 context_idx_map):
        
        assert len(source_questions) == target_labels

        self.contexts = source_contexts
        self.questions = source_questions
        self.target_labels = target_labels

    def __len__(self):
        """
        TODO
        """
        return len(self.source_questions)

    def __getitem__(self, index):
        """
        TODO
        """
        item = {'source_questions': self.source_texts[index],
                'target_labels': self.target_labels[index]}
        return item
    

def load_question_answering_dataset():
    """
    TODO
    """
    # make all the sources, target and context idx dict
    pass


def preprocess_input():
    """
    TODO
    """
    pass


