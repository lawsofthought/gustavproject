"""Gustav: Probabilistic Topic Modelling Toolbox

Usage:
  gustav.py model new <model_name> [--model-type=<model_type>] <data_name> 
  gustav.py model <model_name> update [--iterations=<N>] [--hyperparameters]
  gustav.py data new <data_name> [--data-type=<data_type>] <raw_data_file>
  gustav.py (-h | --help)
  gustav.py --version

Options:
  -h --help                     Show this screen.
  --version                     Show version.
  --iterations=<N>              Model update iterations [default: 100]
  --model-type=<model_type>     Type of topic model [default: hdptm].
  --data-type=<data_type>       Type of data set [default: bag_of_words].

"""
from docopt import docopt
import configobj

try:
    GUSTAV_DATA_ROOT = conf.GUSTAV_DATA_ROOT
except:
    pass

def make_new_model(model_name, model_type, data_name):
    pass

def make_new_data(raw_data_file, data_name, data_type):
    pass


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Gustav 0.0.0')
    print(arguments)