import platform
import os
import shutil
import sys
import subprocess

def main(argv):
    system_type = platform.system()

    traph_build = argv[0]
    traph_root, build_file = os.path.split(traph_build)
    if not os.path.exists(os.path.join(traph_root, "build")):
        os.mkdir("build")
    os.chdir(traph_root + "/build")
    subprocess.run("cmake ../")
    subprocess.run("cmake --build .")

    os.chdir("../")

    if system_type == 'windows':
        shutil.copyfile(os.path.join('build/traph/source/interface/Release/_swig-tensor.pyd'), 'python/pytraph')
    else:
        print('unsupport system')
    
if __name__ == '__main__':
	main(sys.argv)