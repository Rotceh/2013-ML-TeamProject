# 2013 Machine Learning 期末小組戰

## Python Dependencies

Run on Python 3.3.x, with the following dependencies:

- IPython 1.0.0 up, plus its notebook utility
- pandas

主要是這兩個套件，但他們會相依到以下的套件：

- numpy
- scipy

Python 如果怕會影響其他開發環境的話，可以考慮用 virtualenv 來隔離環境。


## R Dependencies

Run on R 3.0.2, with the following dependencies:

- some ML-related packages



## Installation

### Mac

```
brew install python3

# numpy, scipy
brew tap samueljohn/python
brew install numpy --with-python3
brew install scipy --with-python3

# IPython, pandas
easy_install-3.3 ipython[all]
pip3 install pandas
```

```
brew install R
```

### Ubuntu

**Basics**

```
sudo apt-get install build-essential git xz-utils
```

**Python**

```
# Python 3.3.3
sudo apt-get install libncursesw5-dev libreadline-dev libssl-dev libgdbm-dev libc6-dev libsqlite3-dev tk-dev
wget http://python.org/ftp/python/3.3.3/Python-3.3.3.tar.xz
tar Jxvf Python-3.3.3.tar.xz
cd Python-3.3.3
./configure
make
sudo make install

# setuptools
wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
sudo python3 ez_setup.py

# Numpy, Scipy
sudo apt-get install liblapack-dev libatlas-base-dev gfortran
sudo pip-3.3 install nose  # skip this if no need for testing
sudo pip-3.3 install numpy
sudo pip-3.3 install scipy
python3 -c "import numpy as np; np.test()"

# IPython, pandas
sudo easy_install-3.3 ipython[all]
sudo pip-3.3 pandas
```


**R**

Add deb to `sources.list` by `sudo vim /etc/apt/sources.list`

```
## R-related deb
# `raring` should be changed to other name if you are not using Ubuntu 13
deb http://cran.csie.ntu.edu.tw/bin/linux/ubuntu raring/
```

```bash
# add key to sign CRAN packages
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
sudo add-apt-repository ppa:marutter/rdev

sudo apt-get update
sudo apt-get install r-base
```


