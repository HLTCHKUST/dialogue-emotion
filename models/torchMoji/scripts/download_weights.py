from __future__ import print_function
import os
from subprocess import call
from builtins import input

curr_folder = os.path.basename(os.path.normpath(os.getcwd()))

weights_filename = 'pytorch_model.bin'
weights_folder = 'model'
weights_path = '{}/{}'.format(weights_folder, weights_filename)
if curr_folder == 'scripts':
    weights_path = '../' + weights_path
weights_download_link = 'https://www.dropbox.com/s/q8lax9ary32c7t9/pytorch_model.bin?dl=0#'


MB_FACTOR = float(1<<20)

def prompt():
    while True:
        valid = {
            'y': True,
            'ye': True,
            'yes': True,
            'n': False,
            'no': False,
        }
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            print('Please respond with \'y\' or \'n\' (or \'yes\' or \'no\')')

# update with abspath
dirname, filename = os.path.split(os.path.abspath(__file__))

weights_path = dirname + "/../" + weights_path

download = True
if os.path.exists(weights_path):
    print('Weight file already exists at {}. Would you like to redownload it anyway? [y/n]'.format(weights_path))
    download = prompt()
    already_exists = True
else:
    already_exists = False

if download:
    print('About to download the pretrained weights file from {}'.format(weights_download_link))
    if already_exists == False:
        print('The size of the file is roughly 85MB. Continue? [y/n]')
    else:
        os.unlink(weights_path)

    if already_exists or prompt():
        print('Downloading...')

        #urllib.urlretrieve(weights_download_link, weights_path)
        #with open(weights_path,'wb') as f:
        #    f.write(requests.get(weights_download_link).content)

        # downloading using wget due to issues with urlretrieve and requests
        sys_call = 'wget {} -O {}'.format(weights_download_link, os.path.abspath(weights_path))
        print("Running system call: {}".format(sys_call))
        call(sys_call, shell=True)

        if os.path.getsize(weights_path) / MB_FACTOR < 80:
            raise ValueError("Download finished, but the resulting file is too small! " +
                             "It\'s only {} bytes.".format(os.path.getsize(weights_path)))
        print('Downloaded weights to {}'.format(weights_path))
else:
    print('Exiting.')
