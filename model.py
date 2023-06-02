import pytorch_lightning as pl
from transformers import AdamW
from config import config

# General class for summarization
class NewsSummaryModel(pl.LightningModule):
    def __init__(self, model = None):
        super().__init__()
        self.model = config.t5_pretrained_model

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels = None):
        output = self.model(
            input_ids,
            attention_mask = attention_mask,
            labels = labels,
            decoder_attention_mask = decoder_attention_mask
        )

        return output.loss, output.logits

    
    def training_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']


        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
        )

        self.log("train_loss", loss, prog_bar = True, logger = True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

    
        loss, outputs = self(input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            decoder_attention_mask = labels_attention_mask
        )

        self.log("val_loss", loss, prog_bar = True, logger = True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

    
        loss, outputs = self(input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            decoder_attention_mask = labels_attention_mask
        )

        self.log("test_loss", loss, prog_bar = True, logger = True)

        return loss

    def configure_optimizers(self):

        return AdamW(self.parameters(), lr = config.learning_rate)
        

class T5Summarizer:
    def __init__(self):
        pass

    def train(self, preprocessing):
        # Build the T5 model as pytorch-lightning
        t5_model = NewsSummaryModel(config.t5_pretrained_model)

        # Create custom Model Checkpoint
        t5_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath = "t5_checkpoints",
            filename = "t5-best-checkpoint",
            save_top_k = 1,
            verbose = True,
            monitor = "val_loss",
            mode = "min",
        )

        # Create tensorboard logger
        t5_logger = pl.loggers.TensorBoardLogger("t5_lightning_logs", name = "t5-news-summary")

        # Build the trainer
        t5_trainer = pl.Trainer(
            logger = t5_logger,
            checkpoint_callback = t5_checkpoint_callback,
            max_epochs = config.n_epochs,
            gpus = 1,
            progress_bar_refresh_rate = 30
        )

        # Fit the model
        t5_trainer.fit(t5_model, preprocessing.t5_data_module)

        return t5_model