# InternetOfFish2.0


## Installation:

1) follow instructions from https://aiyprojects.withgoogle.com/maker/ for the initial setup.
While we will not use the AIY Maker Kit API directly, using their prebuilt image ensures
that finicky dependencies (like pycoral) are installed correctly
2) enable the camera in raspi-config
3) Clone this repo to your pi using:
```
cd ~
git clone https://github.com/tlancaster6/InternetOfFish2.0
```
4) Open the terminal and install the remaining dependencies. Note building numpy can take
a long time.
```
sudo pip3 install --upgrade pip
sudo pip3 install pandas
sudo pip3 install pause
sudo pip3 install PyYAML
sudo pip3 install sendgrid
sudo pip3 install --upgrade numpy
```

## Rclone Setup for automated uploads (optional)
1) install rclone:
```
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```
2) configure your rclone remote (instructions here: https://rclone.org/dropbox/). The code has only been tested
with Dropbox, but should work with any cloud service supported by rclone
3) To enable automated uploads for a project, first create a new project by running the following command
(but with a unique project id of your choosing)
```
main.py --pid your_project_id
```
4) when prompted, open the config.yaml file, and change the cloud_data_dir variable to your cloud path,
including the rclone remote, where the project will be stored -- e.g., 'my_remote:/my_root/my_project_dir'
5) Save and close the config file. Now rerun main.py (with the same project id ) to initiate data collection.
Automated uploads should now be enabled

## Sendgrid Setup for email notifications (optional)
1) create a free-tier sendgrid account (https://sendgrid.com/en-us) and create a full-access API key 
(https://www.twilio.com/docs/sendgrid/ui/account-and-settings/api-keys).
2) To enable email notifications for a project, first create a new project by running the following command
   (but with a unique project id of your choosing)
```
main.py --pid your_project_id
```
3) when prompted, open the config.yaml file and fill in the following variables:

   - user_email: the email which will receive notifications

   - sendgrid_from_email: the email address you used to set up the sendgrid account

   - sendgrid_api_key: the api key from the previous step

5) Save and close the config file. Now rerun main.py (with the same project id ) to initiate data collection. 
Email notifications should now be enabled.

## Usage
1) open the terminal on the Pi and move into the InternetOfFish2.0 directory
```
cd ~/InternetOfFish2.0
```
2) Generate a new project (replace "your_project_id" with a unique project ID of your choosing)
```
main.py --pid your_project_id
```
3) Once the default config has been generated, the program will exit. This is your opportunity to
edit the config.yaml file with custom values, such as your Sendgrid credentials or dropbox upload 
location (see setup instructions above)


## Troubleshooting

```
Cannot make directory '/var/run/screen': Permission denied
```
To fix this error, run:
```
sudo /etc/init.d/screen-cleanup start
```
