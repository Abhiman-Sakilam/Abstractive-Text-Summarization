from config import config
from model import NewsSummaryModel

class Tester:
    
    def __init__(self, tokenizer = config.t5_tokenizer):
        self.tokenizer = tokenizer
        self.load_model()
    
    def read_input(self, file_path = "./test_article.txt"):
        
        with open(file_path, "r") as f:
            data = f.readlines()
            data = "".join(data)

        return data

    def load_model(self):
        ## Sample predictions using trained summarizer
        model_checkpoint_path = "./model_checkpoint/t5-best-checkpoint.ckpt"
        trained_model = NewsSummaryModel()
        self.model = trained_model.load_from_checkpoint(model_checkpoint_path)
        self.model.freeze()

    def summarize(self, text, trained_model):
        text_encoding = self.tokenizer(
            text, max_length = 512, padding = "max_length", truncation = True, 
            return_attention_mask = True, add_special_tokens = True,
            return_tensors = "pt"
        )

        generated_ids = trained_model.model.generate(
            input_ids = text_encoding['input_ids'],
            attention_mask = text_encoding['attention_mask'],
            max_length = 150,
            num_beams = 2,
            repetition_penalty = 2.5,
            length_penalty = 1.0,
            early_stopping = True
        )

        predictions = [
            self.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
        ]

        return "".join(predictions)

if __name__ == '__main__':
    
    tester = Tester()
    text = tester.read_input(file_path = "./test_article.txt")
    summary = tester.summarize(text, tester.model)
    
    print("\nOriginal Text:\n")
    print(text)

    print("\n\nSummarized Text:\n")
    print(summary)
