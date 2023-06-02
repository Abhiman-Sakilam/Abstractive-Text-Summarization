from preprocessing import Preprocessing
from model import T5Summarizer

if __name__ == '__main__':
    
    # Perform data preprocessing
    preprocessing = Preprocessing()

    # Train T5 Summarizer
    summarizer = T5Summarizer()
    t5_model = summarizer.train(preprocessing)