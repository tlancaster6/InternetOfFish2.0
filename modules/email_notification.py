"""code for managing real-time email notifications"""
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName,
    FileType, Disposition)
import logging
logger = logging.getLogger(__name__)


class Notifier:

    def __init__(self, user_email, api_key, min_notification_interval=600, max_notifications_per_day=20):
        self.user_email, self.api_key = user_email, api_key
        self.min_notification_interval = min_notification_interval
        self.max_notifications_per_day = max_notifications_per_day
        logger.debug('Notifier initialized')

    def send_email(self):
        pass

    def check_conditions(self):
        logger.debug('checking notification conditions')
