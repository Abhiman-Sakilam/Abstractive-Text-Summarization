from data_module import NewsSummaryDataModule
from sklearn.model_selection import train_test_split
from config import config
import pandas as pd

class Preprocessing:
    
    def __init__(self):
        data = self.read_data()
        self.train_df, self.val_df = train_test_split(data, test_size = 0.1)
        self.get_datamodule()

    def read_data(self):
        # Read dataset using pandas
        data = pd.read_csv(config.dataset_path, encoding='latin-1')

        # Select required columns
        data = data[['ctext', 'text']]

        # Rename columns
        data.columns = ['text', 'summary']
        
        # Drop missing rows
        data = data.dropna()

        return data

    # Create datamodule for T5 model
    def get_datamodule(self):
        self.t5_data_module = NewsSummaryDataModule(self.train_df, self.val_df, config.t5_tokenizer, batch_size = config.batch_size)
    