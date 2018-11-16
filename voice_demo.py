import argparse, random, sys, datetime
import numpy as np

import torch
torch.manual_seed(1234)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# PROD side
payment_slots = {
'CardNum'    : ['NUMBER', "What is your card number?"],
'ExpDate'    : ['DATE', "What is the expiration date of the card?"],
'HolderName' : ['PERSON', "What is the name of the card holder?"]
}

billhis_slots = {
'Account'    : ['NUMBER', "What is your account number?"],
'StartDate'       : ['DATE', "What is the start date?"]
}


# ENG side
def demo():
#    slots = payment_slots
#    print ("Can I have your credit card information?")

    slots = billhis_slots
    print ("What can I do for you?")

    values = {}
    while len(values) < len(slots):
        print ("Please input:")
        s = input()
        entities = ner(s)
        next_question = None
        for slot, meta in slots.items():
            if meta[0] in entities: # slot filling (NLU)
                values[slot] = entities[meta[0]][0]
            if slot not in values and next_question is None: # next question (ACTION)
                next_question = meta[1]
        print ("STATE")
        print (values)
        if next_question is not None: print (next_question)
    print ("All slots filled! Update account ... ")



# RES side
checkpoint = None
model = None

def ner(s):
    global checkpoint, model
    if checkpoint is None:
        # Load model
        checkpoint = torch.load('tagger.pt.model')
        # Model creation
        model = TaggerModel(checkpoint['nwords'], checkpoint['nchars'], checkpoint['ntags'], checkpoint['pretrained_list'])
        model.load_state_dict(checkpoint['model_state_dict'])
#        print ("NER model loaded ..")
    words = s.split()
    return do_infer([words], checkpoint['token_to_id'], checkpoint['char_to_id'], checkpoint['id_to_tag'], model)


DIM_EMBEDDING = 100
LSTM_LAYER = 1
LSTM_HIDDEN = 200
CHAR_DIM_EMBEDDING = 30
CHAR_LSTM_HIDDEN = 50
KEEP_PROB = 0.5

class TaggerModel(torch.nn.Module):

    def __init__(self, nwords, nchars, ntags, pretrained_list):
        super().__init__()

        # Create word embeddings
        pretrained_tensor = torch.FloatTensor(pretrained_list)
        self.word_embedding = torch.nn.Embedding.from_pretrained(
                pretrained_tensor, freeze=False)
        # Create input dropout parameter
        self.word_dropout = torch.nn.Dropout(1 - KEEP_PROB)
        # Create LSTM parameters
        self.lstm = torch.nn.LSTM(DIM_EMBEDDING + CHAR_LSTM_HIDDEN, LSTM_HIDDEN, num_layers=LSTM_LAYER,
                batch_first=True, bidirectional=True)
        # Create output dropout parameter
        self.lstm_output_dropout = torch.nn.Dropout(1 - KEEP_PROB)

        # Character-level LSTMs
        self.char_embedding = torch.nn.Embedding(nchars, CHAR_DIM_EMBEDDING)
        self.char_lstm = torch.nn.LSTM(CHAR_DIM_EMBEDDING, CHAR_LSTM_HIDDEN,
                num_layers=1, batch_first=True, bidirectional=False)

        # Create final matrix multiply parameters
        self.hidden_to_tag = torch.nn.Linear(LSTM_HIDDEN * 2, ntags)

    def forward(self, sentences, sent_chars, labels, lengths, char_lengths, cur_batch_size):
        max_length = sentences.size(1)
        max_char_length = sent_chars.size(2)

        # Look up word vectors
        word_vectors = self.word_embedding(sentences)
        # Apply dropout
        dropped_word_vectors = self.word_dropout(word_vectors)

        sent_chars = sent_chars.view(cur_batch_size * max_length, -1)
        char_vectors = self.char_embedding(sent_chars)
        char_lstm_out, (hn, cn) = self.char_lstm(char_vectors, None)
        char_lstm_out = hn[-1].view(cur_batch_size, max_length, CHAR_LSTM_HIDDEN)
        concat_vectors = torch.cat((dropped_word_vectors, char_lstm_out), dim=2)

        # Run the LSTM over the input, reshaping data for efficiency
        packed_words = torch.nn.utils.rnn.pack_padded_sequence(
                concat_vectors, lengths, True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,
                batch_first=True, total_length=max_length)
        # Apply dropout
        lstm_out_dropped = self.lstm_output_dropout(lstm_out)
        # Matrix multiply to get scores for each tag
        output_scores = self.hidden_to_tag(lstm_out_dropped)

        # Calculate loss and predictions
        output_scores = output_scores.view(cur_batch_size * max_length, -1)
        flat_labels = labels.view(cur_batch_size * max_length)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        loss = loss_function(output_scores, flat_labels)
        predicted_tags  = torch.argmax(output_scores, 1)
        predicted_tags = predicted_tags.view(cur_batch_size, max_length)
        return loss, predicted_tags

def do_infer(batch, token_to_id, char_to_id, id_to_tag, model):
    batch.sort(key = lambda x: -len(x))

    # Prepare inputs
    cur_batch_size = len(batch)
    max_length = len(batch[0])
    lengths = [len(v) for v in batch]
    max_char_length = 0
    char_lengths = []
    for tokens in batch:
        for token in tokens:
            max_char_length = max(max_char_length, len(token))
            char_lengths.append(len(token))
        for _ in range(max_length - len(tokens)):
            char_lengths.append(0)
    input_array = torch.zeros((cur_batch_size, max_length)).long()
    input_char_array = torch.zeros((cur_batch_size, max_length, max_char_length)).long()
    output_array = torch.zeros((cur_batch_size, max_length)).long()
    for n, tokens in enumerate(batch):
        token_ids = [token_to_id.get(simplify_token(t), 1) for t in tokens]
        input_array[n, :len(tokens)] = torch.LongTensor(token_ids)
        for m, token in enumerate(tokens):
            char_ids = [char_to_id.get(c, 1) for c in simplify_token(token)]
            input_char_array[n, m,  :len(token)] = torch.LongTensor(char_ids)

    model.to(device)
    # Construct computation
    batch_loss, output = model(input_array.to(device), input_char_array.to(device), output_array.to(device),
            lengths, char_lengths, cur_batch_size)

    # Run computations
    predicted = output.cpu().data.numpy()

    # Update the number of correct tags and total tags
#    for x, y in zip(batch, predicted):
    nes = {}
    prev = 'O'
    ne = ''
    for w, t in zip(batch[0], predicted[0]):
        tag = id_to_tag[t]
#        print (w + ' : ' + tag)
        if tag != 'O': 
            tag = tag[2:]
        if tag == 'CARDINAL' or tag == 'QUANTITY': tag = 'NUMBER'
        if tag == 'GPE': tag = 'LOC'
        if prev == tag:
            ne += ' ' + w
        else:
            if ne != '' and prev != 'O':
                if prev not in nes: nes[prev]  = []
                nes[prev].append(ne)
            ne = w
        prev = tag
    if ne != '' and prev != 'O':
        if prev not in nes: nes[prev]  = []
        nes[prev].append(ne)
    print ("NER extraction results:")
    print (nes)
    return nes

def simplify_token(token):
    chars = []
    for char in token:
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)


if __name__ == '__main__':
    demo()
