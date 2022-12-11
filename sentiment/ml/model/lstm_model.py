import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sentiment.constant.training_pipeline import *
torch.manual_seed(1)


class SentimentLSTM(nn.Module):
    def __init__(self,no_layers: int,vocab_size: int):
        super(SentimentLSTM,self).__init__()
         
        self.output_dim = OUTPUT_DIM
        self.hidden_dim = HIDDEN_DIM
         
        self.no_layers = no_layers
        self.vocab_size = vocab_size
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        
        #lstm
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
                
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        
    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
        
        
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(DEVICE)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(DEVICE)
        hidden = (h0,c0)
        return hidden