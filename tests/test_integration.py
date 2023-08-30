from modules.data_collection import DataCollector
from modules.object_detection import DetectorBase
from modules.upload_automation import Uploader
from modules.behavior_recognition import BehaviorRecognizer
from modules.config_manager import ConfigManager
from modules.email_notification import Notifier


def test_integration(tmp_project):
    config_path = tmp_project

