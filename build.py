import platform
import os
import shutil
import sys
import subprocess

def main(argv):
    system_type = platform.system()
    machine_type = platform.machine()
    is_debug = True

    traph_build = argv[0]
    traph_root, build_file = os.path.split(traph_build)
    if not os.path.exists(os.path.join(traph_root, "build")):
        os.mkdir("build")
    os.chdir(traph_root + "/build")
    
    if machine_type == 'AMD64':
        if is_debug:
            subprocess.run("cmake -DCMAKE_BUILD_TYPE=Debug -G \"Visual Studio 15 2017 Win64\" ../")
        else:
            subprocess.run("cmake -DCMAKE_BUILD_TYPE=Release -G \"Visual Studio 15 2017 Win64\" ../")
        subprocess.run("cmake --build .")
    elif machine_type == 'x86':
        if is_debug:
            subprocess.run("cmake -DCMAKE_BUILD_TYPE=Debug -G \"Visual Studio 15 2017\" ../")
        else:
            subprocess.run("cmake -DCMAKE_BUILD_TYPE=Release -G \"Visual Studio 15 2017\" ../")
        subprocess.run("cmake --build .")
    else:
        print('unsupport machine')

    os.chdir("../")
    
    if system_type == 'Windows':
        if is_debug:
            shutil.copy('build/traph/source/interface/traph_tensor.py', 'python/pytraph/core')
            shutil.copy('build/traph/source/interface/Debug/_traph_tensor.pyd', 'python/pytraph/core')
        else:
            shutil.copy('build/traph/source/interface/traph_tensor.py', 'python/pytraph/core')
            shutil.copy('build/traph/source/interface/Release/_traph_tensor.pyd', 'python/pytraph/core')
    else:
        print('unsupport system')
    
if __name__ == '__main__':
	main(sys.argv)