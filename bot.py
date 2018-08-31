from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from configparser import ConfigParser

from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.channels import HttpInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter

from rasa_slack_connector import SlackInput
from rasa_core.channels.facebook import FacebookInput

logger = logging.getLogger(__name__)

def train_nlu():
    training_data = load_data('data/nlu_intents/')
    trainer = Trainer(config.load("nlu_model_config.yml"))
    trainer.train(training_data)
    model_directory = trainer.persist('./models/nlu', fixed_model_name='intents')
    return model_directory

def train_diag(domain_file="dynamo_domain.yml",
                          training_data_file='data/dialouge_stories/stories.md',
                          model_path='models/dialouge'):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(max_history=2), KerasPolicy()])

    training_data = agent.load_data(training_data_file)

    agent.train(training_data,
                augmentation_factor= 50,
                epochs = 500,
                batch_size = 50,
                validation_split = 0.2)

    agent.persist(model_path)

    return agent

def train_diag_model_online(input_channel, interpreter,
                          domain_file="dynamo_domain.yml",
                          training_data_file='data/dialouge_stories/stories.md'):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(max_history=2), KerasPolicy()],
                  interpreter=interpreter)

    training_data = agent.load_data(training_data_file)
    agent.train_online(training_data,
                       input_channel=input_channel,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)

    return agent


def run_bot(slack_params, console = False):
    nlu_interpreter = RasaNLUInterpreter('models/nlu/default/intents')
    agent = Agent.load('models/dialouge', interpreter=nlu_interpreter)

    if console == True:
        agent.handle_channel(ConsoleInputChannel())
    else:
        input_channel = SlackInput(slack_params['slack_dev_token'],
                                    slack_params['slack_client_token'],
                                    slack_params['verification_token'],
                                    True)
        agent.handle_channel(HttpInputChannel(5004, "/", input_channel))

    return agent

methods = {'train_nlu':train_nlu, 'train_diag':train_diag, 'run_bot':run_bot}

if __name__ == '__main__':
    utils.configure_colored_logging(loglevel="INFO")
    run_config = ConfigParser()
    run_config.read('bot_config.ini')
    run_mode = run_config.get('params','mode')
    slack_params = {'slack_dev_token' : run_config.get('connectors','slack_dev_token'),
                    'slack_client_token' : run_config.get('connectors','slack_client_token'),
                    'verification_token' : run_config.get('connectors','verification_token')}

    if run_mode == 'run_bot':
        methods[run_mode](slack_params, console = False)
    else:
        methods[run_mode]()
