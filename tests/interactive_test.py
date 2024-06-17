from modules.data_collection import DataCollector
from modules.object_detection import DetectorBase
from modules.upload_automation import Uploader
from modules.behavior_recognition import BehaviorRecognizer
from modules.config_manager import ConfigManager
from modules.email_notification import Notifier
from main import MODEL_DIR, DEFAULT_DATA_DIR
from tests.mocks import MockDataCollector
from tests.conftest import TESTING_RESOURCE_DIR


def run_interactive_test():
    print('Commencing interactive test \n\n')
    print('generating default config')

    config_path = DEFAULT_DATA_DIR / 'interactive_test' / 'config.yaml'
    config_manager = ConfigManager(config_path)
    config_manager.generate_new_config()

    print('default config generated. To test email notification and automated uploads, open the config.yaml file now\n'
          '(located under projects/interactive_test), provide appropriate values for cloud_data_dir, user_email, \n'
          'sendgrid_api_key, and sendgrid_from_email, then save and close the config file. Or leave the config.yaml\n'
          'file alone to skip these tests. ')
    input('Press enter when ready to resume testing')

    print('')





