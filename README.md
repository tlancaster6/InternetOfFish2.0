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

5) Move into the repo root directory and run pytest to check that everything is working
```
cd ~/InternetOfFish2.0
pytest
```

## Rclone Setup for automated uploads (optional)

## Sendgrid Setup for email notifications (optional)

## Usage
1) open the terminal on the Pi and move into the InternetOfFish2.0 directory
```
cd ~/InternetOfFish2.0
```

