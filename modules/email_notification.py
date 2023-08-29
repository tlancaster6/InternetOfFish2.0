"""code for managing real-time email notifications"""
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName,
    FileType, Disposition)


class Notifier:

    def __init__(self, user_email, api_key):
        self.user_email, self.api_key = user_email, api_key

    def send_email(self):
        pass
