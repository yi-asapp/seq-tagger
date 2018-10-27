import argparse, random, sys, datetime
import numpy as np

from crf2 import CRF

PAD = "__PAD__"
UNK = "__UNK__"
DIM_EMBEDDING = 100
LSTM_LAYER = 2
LSTM_HIDDEN = 250
BATCH_SIZE = 10
LEARNING_RATE = 0.015
LEARNING_DECAY_RATE = 0.05
EPOCHS = 100
KEEP_PROB = 0.5
GLOVE = "./data/glove.6B.100d.txt"
WEIGHT_DECAY = 1e-8

import torch
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from metric import get_ner_fmeasure

# Data reading
def read_data(filename):
    content = []
    with open(filename, 'r') as data_src:
        tokens, tags = [], []
        for line in data_src:
            if line.startswith("-DOCSTART-"): continue
            parts = line.strip().split()
            if len(parts) < 2:
                if len(tokens) > 0:
                    content.append((tokens, tags))
                    tokens, tags = [], []
                continue
            tokens.append(parts[0])
            tags.append(parts[-1])
    return content

def simplify_token(token):
    chars = []
    for char in token:
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)

def main():
    parser = argparse.ArgumentParser(description='NER tagger.')
    parser.add_argument('training_data')
    parser.add_argument('dev_data')
    parser.add_argument('test_data')
    args = parser.parse_args()

    train = read_data(args.training_data)
    dev = read_data(args.dev_data)
    test = read_data(args.test_data)

    # Make indices
    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = [PAD]
    tag_to_id = {PAD: 0}
    for tokens, tags in train + dev + test:
        for token in tokens:
            token = simplify_token(token)
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
                id_to_token.append(token)
        for tag in tags:
            if tag not in tag_to_id:
                tag_to_id[tag] = len(tag_to_id)
                id_to_tag.append(tag)
    id_to_tag += ["START_TAG", "END_TAG"]
    NWORDS = len(token_to_id)
    NTAGS = len(tag_to_id)

    # Load pre-trained GloVe vectors
    pretrained = {}
    for line in open(GLOVE):
        parts = line.strip().split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector
    pretrained_list = []
    scale = np.sqrt(3.0 / DIM_EMBEDDING)
    for word in id_to_token:
        # apply lower() because all GloVe vectors are for lowercase words
        if word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
            pretrained_list.append(random_vector)

    # Model creation
    model = TaggerModel(NWORDS, NTAGS, pretrained_list)

    # Create optimizer and configure the learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY)
    rescale_lr = lambda epoch: 1 / (1 + LEARNING_DECAY_RATE * epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda=rescale_lr)

    dev_f1, test_f1 = 0, 0
    expressions = (model, optimizer)
    for epoch in range(EPOCHS):
        random.shuffle(train)

        # Update learning rate
        scheduler.step()

        model.train()
        model.zero_grad()
        loss, trf1 = do_pass(train, token_to_id, tag_to_id, id_to_tag, expressions,
                True, epoch)

        model.eval()
        _, df1 = do_pass(dev, token_to_id, tag_to_id, id_to_tag, expressions, False, epoch)
        _, tef1 = do_pass(test, token_to_id, tag_to_id, id_to_tag, expressions, False, epoch)
        if df1 > dev_f1:
            dev_f1, test_f1 = df1, tef1
        print("{0} {1} loss {2} dev-f1 {3:.4f} test-f1 {4:.4f}".format(datetime.datetime.now(),
                epoch, loss, df1, tef1))

    print("Finish training - dev-f1 {0:.4f} test-f1 {1:.4f}".format(dev_f1, test_f1))

    # Save model
    #torch.save(model.state_dict(), "tagger.pt.model")

    # Load model
    #model.load_state_dict(torch.load('tagger.pt.model'))


class TaggerModel(torch.nn.Module):

    def __init__(self, nwords, ntags, pretrained_list):
        super().__init__()

        # Create word embeddings
        pretrained_tensor = torch.FloatTensor(pretrained_list)
        self.word_embedding = torch.nn.Embedding.from_pretrained(
                pretrained_tensor, freeze=False)
        # Create input dropout parameter
        self.word_dropout = torch.nn.Dropout(1 - KEEP_PROB)
        # Create LSTM parameters
        self.lstm = torch.nn.LSTM(DIM_EMBEDDING, LSTM_HIDDEN, num_layers=LSTM_LAYER,
                batch_first=True, bidirectional=True)
        # Create output dropout parameter
        self.lstm_output_dropout = torch.nn.Dropout(1 - KEEP_PROB)
        # Create final matrix multiply parameters
        self.hidden_to_tag = torch.nn.Linear(LSTM_HIDDEN * 2, ntags + 2)

        self.crf = CRF(tagset_size=ntags, True)

    def forward(self, sentences, mask, labels, lengths, cur_batch_size, epo):
        max_length = sentences.size(1)

        # Look up word vectors
        word_vectors = self.word_embedding(sentences)
        # Apply dropout
        dropped_word_vectors = self.word_dropout(word_vectors)
        # Run the LSTM over the input, reshaping data for efficiency
        packed_words = torch.nn.utils.rnn.pack_padded_sequence(
                dropped_word_vectors, lengths, True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,
                batch_first=True, total_length=max_length)
        # Apply dropout
        lstm_out_dropped = self.lstm_output_dropout(lstm_out)
        # Matrix multiply to get scores for each tag
        output_scores = self.hidden_to_tag(lstm_out_dropped)

        if epo > 15:
            loss = self.crf.neg_log_likelihood_loss(output_scores, mask, labels)
            _, predicted_tags = self.crf(output_scores, mask)
        else:
        # Calculate loss and predictions
            output_scores = output_scores.view(cur_batch_size * max_length, -1)
            flat_labels = labels.view(cur_batch_size * max_length)
            loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
            loss = loss_function(output_scores, flat_labels)
            predicted_tags  = torch.argmax(output_scores, 1)

        predicted_tags = predicted_tags.view(cur_batch_size, max_length)
        return loss, predicted_tags

def do_pass(data, token_to_id, tag_to_id, id_to_tag, expressions, train, epo):
    model, optimizer = expressions

    # Loop over batches
    loss = 0
    gold_lists, pred_lists = [], []
    for start in range(0, len(data), BATCH_SIZE):
        batch = data[start : start + BATCH_SIZE]
        batch.sort(key = lambda x: -len(x[0]))

        # Prepare inputs
        cur_batch_size = len(batch)
        max_length = len(batch[0][0])
        lengths = [len(v[0]) for v in batch]
        input_array = torch.zeros((cur_batch_size, max_length)).long()
        mask_array = torch.zeros((cur_batch_size, max_length)).byte()
        output_array = torch.zeros((cur_batch_size, max_length)).long()
        for n, (tokens, tags) in enumerate(batch):
            token_ids = [token_to_id.get(simplify_token(t), 1) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]
            mask_ids = [1 for t in tags]
            input_array[n, :len(tokens)] = torch.LongTensor(token_ids)
            mask_array[n, :len(tokens)] = torch.LongTensor(mask_ids)
            output_array[n, :len(tags)] = torch.LongTensor(tag_ids)

        model.to(device)
        # Construct computation
        batch_loss, output = model(input_array.to(device), mask_array.to(device),
                output_array.to(device), lengths, cur_batch_size, epo)

        # Run computations
        if train:
            batch_loss.backward()
            optimizer.step()
            model.zero_grad()
            loss += batch_loss.item()
        predicted = output.cpu().data.numpy()

        for (_, g), a in zip(batch, predicted):
            gold_list, pred_list = [], []
            for gt, at in zip(g, a):
                at = id_to_tag[at]
                gold_list.append(gt)
                pred_list.append(at)
            gold_lists.append(gold_list)
            pred_lists.append(pred_list)

    return loss, get_ner_fmeasure(gold_lists, pred_lists)[-1]

if __name__ == '__main__':
    main()
