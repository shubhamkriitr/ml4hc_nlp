from torch import nn
from torch import functional as F
from torch.optim.adam import Adam
from sklearn.metrics import accuracy_score, f1_score
from util import logger
import torch
# # TODO
# >>> self.model.set_embeddings(embeddings)
# >>> self.model.set_word_to_index(word_to_index)
# >>> self.model.set_index_to_word(index_to_word)
class CnnWithResidualConnection(nn.Module):

    def __init__(self, config={"num_classes": 5}, *args, **kwargs) -> None:
        super().__init__()
        self.num_classes = config["num_classes"]
        self.embeddings = None
        self.word_to_index = None
        self.index_to_word = None
        

        self.expand_channel = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU())
        
        self.block_0 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU()
        )


        self.downsample_0 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )

        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )

        self.downsample_1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=16*2, out_channels=32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*2),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv1d(in_channels=32*2, out_channels=32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32*2, out_channels=32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*2),
            nn.ReLU()
        )

        self.downsample_2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=16*4, out_channels=32*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32*4),
            nn.ReLU()
        )

        
        self.flatten = nn.Flatten()

        self.fc_block = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.num_classes),
            nn.ReLU()
        )
        self.last_layer_activation = nn.Softmax(dim=1)

        #generic operations
        self.relu = nn.ReLU()

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
        x = torch.unsqueeze(x, dim=1) # adding a channel dimension
        
        output_ = self.expand_channel(x)
        identity_output_ = output_

        output_ = self.block_0(output_)
        output_ = identity_output_ + output_ # shortcut connection
        output_ = self.downsample_0(output_)
        identity_output_ = output_

        output_ = self.block_1(output_)
        output_ = identity_output_ + output_ # shortcut connection
        output_ = self.downsample_1(output_)
        identity_output_ = output_

        output_ = self.block_2(output_)
        output_ = identity_output_ + output_ # shortcut connection
        output_ = self.downsample_2(output_)

        # Average pooling
        output_ = torch.mean(output_, dim=2) # ouput --> (B, 128)
       
        output_ = self.flatten(output_)

        output_ = self.fc_block(output_)

        

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
    

        


