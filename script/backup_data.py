

from shutil import *
import os

backup_types = ['.log', '.data', '.save', '.info']

def backup():
    print("Back up files in dir: " + os.getcwd())

    for root, dirs, files in os.walk(os.getcwd()):
        for f in files:
             # print(f + " " + str(os.path.splitext(f)))
            if os.path.splitext(f)[-1] in backup_types:
                print("Copy " + f + " to " + f +".bak")
                copy2(f, f+'.bak')
                


if __name__ == "__main__":
    backup()


