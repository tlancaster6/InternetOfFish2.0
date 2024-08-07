from main import Runner, DEFAULT_DATA_DIR
from datetime import datetime, timedelta
import logging
import sys

logger = logging.getLogger(__name__)

pid = input('enter pid: ')
config_path = DEFAULT_DATA_DIR / pid / 'config.yaml'
runner = Runner(config_path)
runner.video_split_interval = timedelta(minutes=1)
mode_switch_interval = timedelta(minutes=3)

for test_iter in range(3):
    logger.info(f'running test iteration {test_iter}')
    runner.end_time = (datetime.now() + mode_switch_interval).time()
    runner.active_mode(round_video_split_time=False)
    runner.start_time = (datetime.now() + mode_switch_interval).time()
    runner.end_time = (datetime.now() + mode_switch_interval + mode_switch_interval).time()
    runner.passive_mode()

try:
    raise Exception('test exception')
except Exception as e:
    logger.warning(f'unknown exception: {e}')
    notification = Notification(subject=f'Unexpected Error in {runner.config.project_id}',
                                message=f'{e}',
                                attachment_path=log_path)
    runner.notifier.send_user_email(notification)
    runner.notifier.send_admin_email(notification)
    try:
        runner.collector.shutdown()
        runner.uploader.convert_and_upload()
    finally:
        sys.exit(0)
