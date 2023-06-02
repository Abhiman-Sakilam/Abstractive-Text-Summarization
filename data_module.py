import pytorch_lightning as pl
import pandas as pd
from config import config
from dataset import NewsSummaryDataset
from torch.utils.data import DataLoader

class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_df: pd.DataFrame,
                 val_df : pd.DataFrame,
                 tokenizer,
                 batch_size: int = config.batch_size,
                 text_max_token_len : int = config.text_token_max_length,
                 summary_max_token_len : int = config.summary_token_max_length
                 ):

        super().__init__()

        self.train_df = train_df
        self.test_df = val_df

        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage = None):

        # Build train dataset from custom dataset
        self.train_dataset = NewsSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

        # Build validation dataset from custom dataset
        self.test_dataset = NewsSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        # Convert dataset to dataloader and return
        return DataLoader(self.train_dataset,
                          batch_size = self.batch_size,
                          shuffle = True,
                          num_workers = config.num_workers)
        
    def val_dataloader(self):
        # Convert dataset to dataloader and return
        return DataLoader(self.test_dataset,
                          batch_size = self.batch_size,
                          shuffle = False,
                          num_workers = config.num_workers)
        
    def test_dataloader(self):
        # Convert dataset to dataloader and return
        return DataLoader(self.test_dataset,
                          batch_size = self.batch_size,
                          shuffle = False,
                          num_workers = config.num_workers)