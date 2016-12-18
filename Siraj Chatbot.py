'''
This chat bot uses Siraj's tutoral (https://www.youtube.com/watch?v=SJDEOWLHYVo) and the cornel data set https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

the dataset is unzipped here ~/sashweed/moviedata/cornel movie-dialog corpus

lets see if this works ---- well its not going to
'''

import tensorflow as tf
import data_utils
import seq2seq_model

def train():
    #prepare dataset
    enc_train, dec_train = data_utils.prepare.custom.data(
        gconfig['working_directory'])

    train_set = read data(enc_train, dec_train)

#Config Seq2Seq
def seq2seq_f(encoder_inputs, decoder_inputs, do_decode)
    return tf.nn.seq2seq.embeding_attention_seq2seq(
        encoder_inputs, decoder_inputs, cell,
        num_encoder_symbols=source_vocab_size,
        num_decoder_symbols=target_vocab_size,
        embeding_size=size,
        output_projection=output_projection,
        feed_previous=do_decode)

with tf.session(config=config) as sess:
    #create model
    model = create_model(sess, False)

    while True
        sess.run(model)

        #save checkpoint and zero timer and loss
        checkpoint_point = os.path.join(gconfig['working_directory'], "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_point, global_step=model.global_step)

