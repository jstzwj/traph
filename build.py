import platform
import os
import shutil
import sys
import subprocess

def main(argv):
    traph_build = argv[0]
    traph_root, build_file = os.path.split(traph_build)
    if not os.path.exists(traph_root + "/build"):
        os.mkdir("build")
    os.chdir(traph_root + "/build")
    subprocess.run("cmake ../")
    
if __name__ == '__main__':
	main(sys.argv)