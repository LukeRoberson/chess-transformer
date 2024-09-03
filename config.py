'''
Read the configuration file
Load settings for the model
'''

import yaml


class Config:
    '''
    Class to manage the configuration of the model
    '''

    def __init__(
        self,
        config_file: str = 'config.yaml',
    ) -> None:
        '''
        Constructor
        Read the config file
        '''

        # Read the configuration file
        self.config_file = config_file
        with open(config_file, 'r') as file:
            settings = yaml.safe_load(file)

        # Dataset settings
        self.dataset = {}
        self.dataset['path'] = (
            settings['dataset']['path']
        )
        self.dataset['batch_size'] = (
            settings['dataset']['batch_size']
        )
        self.dataset['test_split'] = (
            settings['dataset']['test_split']
        )
        self.dataset['chunk_percent'] = (
            settings['dataset']['chunk_percent']
        )

        # Model settings
        self.model = {}
        self.model['block_size'] = (
            settings['model']['block_size']
        )
        self.model['embedding_size'] = (
            settings['model']['embedding_size']
        )
        self.model['heads'] = (
            settings['model']['heads']
        )
        self.model['layers'] = (
            settings['model']['layers']
        )

        # Regularization settings
        self.regularization = {}
        self.regularization['dropout'] = (
            settings['regularization']['dropout']
        )
        self.regularization['weight_decay'] = (
            settings['regularization']['weight_decay']
        )

        # Training settings
        self.training = {}
        self.training['epochs'] = (
            settings['train']['epochs']
        )
        self.training['learning_rate'] = (
            settings['train']['learning_rate']
        )
        self.training['warmup_steps'] = (
            settings['train']['warmup_steps']
        )
        self.training['eval_iterations'] = (
            settings['train']['eval_iterations']
        )
        self.training['checkpoint'] = (
            settings['train']['checkpoint']
        )
        self.training['resume'] = (
            settings['train']['resume']
        )

        # Scheduler settings
        self.scheduler = {}
        self.scheduler['first_cycle'] = (
            settings['scheduler']['first_cycle']
        )
        self.scheduler['cycle_factor'] = (
            settings['scheduler']['cycle_factor']
        )
        self.scheduler['min_lr'] = (
            settings['scheduler']['min_lr']
        )
