import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # load a pre-trained CNN
        resnet = models.resnet50(pretrained=True)
        # don't compute gradients for the CNN parameters since we don't want to update them during training
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # remove the classification layer from the model
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # add an embedding layer at the end
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        # reshape features to make it compatible with the embedding layer's input requirements
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # embed the captions to the size same as that used for cnn output
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # we want output with second dimension same as captions.shape[1] (i.e. sequence length)
        # since we are going to concat cnn features with captions, the second dimension will increase by 1
        # so, we remove the end token from input caption
        captions = captions[:, :-1]  
        # embed captions
        embedded_captions = self.embed(captions)
        # concat cnn features and embedded captions
        lstm_in = torch.cat((features.unsqueeze(dim=1), embedded_captions), 1)
        # pass through lstm
        lstm_out, _ = self.lstm(lstm_in)
        # pass through FC
        out = self.fc(lstm_out)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # to store predicted caption's word indices
        caption_ind = []
        lstm_in = inputs
        
        # predict the first word; should always be <start> token
        lstm_out, states = self.lstm(lstm_in, states)
        fc_out = self.fc(lstm_out)
        # get the index of the word having max probability
        out = torch.argmax(fc_out)
        caption_ind.append(out.item())
        
        # predicted remaining words in the caption
        for i in range(1, max_len):
            lstm_in = self.embed(out)
            lstm_in = lstm_in.view(1, 1, -1)
            lstm_out, states = self.lstm(lstm_in, states)
            fc_out = self.fc(lstm_out)
            # get the index of the word having max probability
            out = torch.argmax(fc_out)
            caption_ind.append(out.item())
            # if <end> token is predicted, stop prediction process
            if out.item() == 1:
                break
        
        return caption_ind