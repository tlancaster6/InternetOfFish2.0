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
4) Install remaining dependencies:
```
sudo pip3 install -U pytest
sudo pip3 install pandas
sudo pip3 install pause
sudo pip3 install PyYAML
sudo pip3 install sendgrid
```

5) Move into the repo root directory and run pytest to run a basic battery of tests
```
cd ~/InternetOfFish2.0
pytest
```

## Rclone Setup for automated uploads (optional)
1) install rclone:
```
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```
2) configure your rclone remote (instructions here: https://rclone.org/dropbox/)
3) To enable automated uploads for a project, first create a new project by running the following command
   (but with a unique project id of your choosing)
```commandline
main.py --pid your_project_id
```
4) when prompted, open the config.yaml file, and change the cloud_data_dir variable to your cloud path,
including the rclone remote, where the project will be stored.
5) rerun main.py (with the same project id )

## Sendgrid Setup for email notifications (optional)

## Usage
1) open the terminal on the Pi and move into the InternetOfFish2.0 directory
```
cd ~/InternetOfFish2.0
```

