InternetOfFish2.0


Installation:
1) follow instructions from https://aiyprojects.withgoogle.com/maker/ for the initial setup.
While we will not use the AIY Maker Kit API directly, using their prebuilt image ensures
that finicky dependencies (like pycoral) are installed correctly
2) enable the camera in raspi-config
3) Clone this repo to your pi using:
```
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
