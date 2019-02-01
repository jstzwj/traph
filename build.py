import platform
import os
import shutil
import sys
import subprocess

def main(argv):
    traph_root = argv[0]
    subprocess.run("cmake .")
    subprocess.run("cmake .")
if __name__ == '__main__':
	main(sys.argv)