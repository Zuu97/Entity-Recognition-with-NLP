data_path = 'data/ner_dataset.csv'
oov_token = '<OOV>'
max_length = 80
cutoff = 0.1

all_labels = ['<OOV>', 'I-nat', 'I-geo', 'B-per', 'I-per', 'I-gpe', 'B-geo', 'B-gpe', 'B-tim', 'B-art', 'I-org', 'I-art', 'B-eve', 'B-nat', 'I-tim', 'B-org', 'I-eve', 'O']
label_dict = {label : i for i, label in enumerate(all_labels)}

embedding_dimS = 512
trunc_type = 'post'
epochs_rnn = 20
batch_size_rnn = 128
size_lstm  = 128
dense_1_rnn = 256
dense_2_rnn = 64
learning_rate = 0.001
rnn_weights = "weights/dog_lstm.h5"
rnn_architecture = "weights/dog_lstm.json"