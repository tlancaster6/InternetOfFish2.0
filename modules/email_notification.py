"""code for managing real-time email notifications"""
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName,
    FileType, Disposition)
import logging
logger = logging.getLogger(__name__)
class Notification:
    def __init__(self, msg_src, msg_type, msg, attachment_path):
        self.msg_src, self.msg_type, self.msg, self.attachment_path = msg_src, msg_type, msg, attachment_path
        self.id = gen_utils.current_time_iso()

    def __str__(self):
        return (f"time: {self.id}\n"
                f"source: {self.msg_src}\n"
                f"type: {self.msg_type}\n"
                f"message: {self.msg}\n"
                f"attachment: {self.attachment_path}")

    def timestamp(self):
        return dt.fromisoformat(self.id).timestamp()


class Notifier:

    def __init__(self, user_email, api_key, min_notification_interval=600, max_notifications_per_day=20):
        logger.debug()
        self.user_email, self.api_key = user_email, api_key
        self.min_notification_interval = min_notification_interval
        self.max_notifications_per_day = max_notifications_per_day
        logger.debug('Notifier initialized')

    def send_email(self):
        pass
