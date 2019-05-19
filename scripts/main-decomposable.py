import ConfigParser
import argparse
import os

from gtd.log import set_log_level
from gtd.ml.utils import TensorBoardLogger
from gtd.utils import Config

from strongsup.decoder import Decoder
from strongsup.domain import get_domain
from strongsup.embeddings import GloveEmbeddings

set_log_level('DEBUG')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('exp_id', nargs='+')
args = arg_parser.parse_args()
exp_id = args.exp_id


glove_embeddings = GloveEmbeddings(vocab_size=20000)
config = Config.from_file(exp_id[0])
domain = get_domain(config)
utterance_length = config.parse_model.utterance_embedder.utterance_length
utterance_num = config.parse_model.utterance_embedder.utterance_num
iterations_per_utterance = config.decoder.train_exploration_policy.iterations_per_utterance
tb_logger = TensorBoardLogger(os.path.join(os.getcwd(), 'data', 'decomposable', 'tensorboard'))
decomposable_weights_file = os.path.join(os.getcwd(), 'data', 'decomposable', 'decomposable_weights_{}.hdf5')
decomposable_config_file = os.path.join(os.getcwd(), 'data', 'decomposable', 'decomposable_config.ini')

deco_config = ConfigParser.ConfigParser()
deco_config.read(decomposable_config_file)
deco_config = dict(deco_config.items('Params'))
deco_config['lr'] = float(deco_config['lr'])
deco_config['dropout'] = float(deco_config['dropout'])
deco_config['hidden_layers'] = int(deco_config['hidden_layers'])

decoder = Decoder(None, config.decoder, domain, glove_embeddings, domain.fixed_predicates,
                  utterance_length * utterance_num,
                  iterations_per_utterance * utterance_num,
                  tb_logger,
                  decomposable_weights_file=decomposable_weights_file,
                  decomposable_config=deco_config
                  )

if deco_config['weights_for_epoch']:
    csv_file = os.path.join(os.getcwd(), 'data', 'decomposable', 'test-decomposable.csv')
else:
    csv_file = os.path.join(os.getcwd(), 'data', 'decomposable', 'train-decomposable.csv')
decoder.decomposable_from_csv(csv_file)
