from main import Runner, DEFAULT_DATA_DIR
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

pid = input('enter pid')
config_path = DEFAULT_DATA_DIR / pid / 'config.yaml'
runner = Runner(config_path)
runner.video_split_interval = timedelta(minutes=2)
mode_switch_interval = timedelta(minutes=5)

for test_iter in range(3):
    logger.info(f'running test iteration {test_iter}')
    runner.end_time = (datetime.now() + mode_switch_interval).time()
    runner.active_mode(round_video_split_time=False)
    runner.start_time = datetime.now().time()
    runner.passive_mode()
