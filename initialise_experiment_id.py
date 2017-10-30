import argparse 
from collections import defaultdict
from sigopt import Connection
import time

parser = argparse.ArgumentParser(description='creating sigopt experiment id')
parser.add_argument('-parameters', default='default', help='which set of model training parameters to use')
parser.add_argument('-model_type', default='baseline', help='which type of data preprocessing is being used')
opts = parser.parse_args()

def main():
    dict_of_list_of_dicts_of_parameters = \
        {
            'baseline':
                [
                    {
                        'name': '-word_vec_dim',
                        'type': 'int',
                        'bounds': 
                        {
                            'min': 100.0,
                            'max': 1000.0
                        }
                    }
                ],
            'layers,dropout,vec_sizes':
                [
                    {
                        'name': '-dropout',
                        'type': 'double',
                        'bounds': {
                            'min': 0.1,
                            'max': 0.5
                        }
                    },
                    {
                        'name': '-rnn_size',
                        'type': 'int',
                        'bounds': {
                            'min': 128,
                            'max': 1024
                        }
                    },
                    {
                        'name': '-word_vec_size',
                        'type': 'int',
                        'bounds': {
                            'min': 128,
                            'max': 1024
                        }
                    },
                    {
                        'name': '-layers',
                        'type': 'int',
                        'bounds': {
                            'min': 1,
                            'max': 3
                        }                    
                    }
                ]
        }

    test_dict = defaultdict(lambda:'lol', dict_of_list_of_dicts_of_parameters)
    # get the set of parameters from the dict using the model name
    # we really need to rename this stuff
    model_parameters = test_dict[opts.parameters]

    if model_parameters == 'lol':
        print('27604')
        quit()

    now = time.strftime('%Y_%m_%d__%H_%M')
    conn = Connection(client_token="IFAQZABYDOBABXMSZYAWSYKHYSONHNPEACATCSCCIDXDQFLG")
    experiment = conn.experiments().create(
        name='_'.join([opts.model_type, opts.parameters, now]),
        parameters=model_parameters
    )

    print(experiment.id)

if __name__ == '__main__':
    main()