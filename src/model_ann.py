from torch import nn
import torch

class FullyConnectedModel(nn.Module):

    def __init__(self, config={"num_classes": 5, "vector_size": 200}, *args, **kwargs) -> None:
        super().__init__()
        self.num_classes = config["num_classes"]
        self.input_vector_size = config["vector_size"]
        self.embeddings = None
        self.word_to_index = None
        self.index_to_word = None
        
        self.flatten = nn.Flatten()

        self.fc_block = nn.Sequential(
            nn.Linear(in_features=self.input_vector_size, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=self.num_classes),
            nn.ReLU()
        )
        self.last_layer_activation = nn.Softmax(dim=1)



        self.initialize_parameters()
    
    def set_embeddings(self, embeddings):
        self.embeddings = nn.EmbeddingBag.from_pretrained(embeddings,
                                                          freeze=True)
    
    def set_word_to_index(self, word_to_index):
        self.word_to_index = word_to_index
    
    def set_index_to_word(self, index_to_word):
        self.index_to_word = index_to_word
        

    def forward(self, x):
        # x will be a tuple (word indices, offsets)
        text, offsets = x
        x = self.embeddings(text, offsets)
        # x would be now of size (batch_size, num_embedding_dims)
        x = self.flatten(x)
        output_ = self.fc_block(x)

        

        return output_ # this is logit not probability
    
    def predict(self, x):
        output_ = self.forward(x)
        output_ = self.last_layer_activation(output_)
        return output_
    
    def initialize_parameters(self):
        for idx, m in enumerate(self.modules()):
            # print(idx, '->', m)
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                # print("ConvLayer or LinearLayer")
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                # print("BatchNormLayer")
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

        
class FullyConnectedModel300(FullyConnectedModel):
    # embedding vector size 300 expected
    def __init__(self, config={ "num_classes": 5,"vector_size": 300 },
                 *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

class FullyConnectedModelWithDropout(FullyConnectedModel):
    def __init__(self, config={ "num_classes": 5 }, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.fc_block = self.fc_block = nn.Sequential(
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=100, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=16, out_features=self.num_classes),
            nn.ReLU()
        )
        
class FullyConnectedModelUnfrozenEmdeddings(FullyConnectedModel):
    def __init__(self, config={ "num_classes": 5 , "vector_size": 200}, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
    
    def set_embeddings(self, embeddings):
        self.embeddings = nn.EmbeddingBag.from_pretrained(embeddings,
                                                          freeze=False)