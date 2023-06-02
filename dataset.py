from torch.utils.data.dataset import Dataset
from config import config
from torch.utils.data import Dataset
import pandas as pd

class NewsSummaryDataset(Dataset):
    def __init__(self, 
                 data : pd.DataFrame,
                 tokenizer,
                 text_max_token_len : int = config.text_token_max_length,
                 summary_max_token_len : int = config.summary_token_max_length):
      
      self.tokenizer = tokenizer
      self.data = data
      self.text_max_token_len = text_max_token_len
      self.summary_max_token_len = summary_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index : int):
        data_row = self.data.iloc[index]
        # Encode text
        text = data_row['text']
        text_encoding = self.tokenizer(text,
                                      max_length = self.text_max_token_len,
                                      padding = "max_length",
                                      truncation = True,
                                      return_attention_mask = True,
                                      add_special_tokens = True,
                                      return_tensors = "pt")

        
        
        # Encode summary
        summary = data_row['summary']
        summary_encoding = self.tokenizer(summary,
                                      max_length = self.summary_max_token_len,
                                      padding = "max_length",
                                      truncation = True,
                                      return_attention_mask = True,
                                      add_special_tokens = True,
                                      return_tensors = "pt")
        
        # Replace 0's with -100 to let the transformer understand
        labels = summary_encoding['input_ids']
        labels[labels == 0] = -100

        return dict(
            text = text,
            summary = summary,
            text_input_ids = text_encoding['input_ids'].flatten(),
            text_attention_mask = text_encoding['attention_mask'].flatten(),
            labels = labels.flatten(),
            labels_attention_mask = summary_encoding['attention_mask'].flatten()
        )