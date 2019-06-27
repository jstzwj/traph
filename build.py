import platform
import os
import shutil
import sys
import subprocess

def main(argv):
    system_type = platform.system()
    machine_type = platform.machine()
    is_debug = False

    traph_build_script = os.path.abspath(sys.argv[0])
    traph_root, build_file = os.path.split(traph_build_script)
    if not os.path.exists(os.path.join(traph_root, "build")):
        os.mkdir("build")
    os.chdir(traph_root + "/build")

    if system_type == 'Windows':
        if machine_type == 'x86_64':
            # -DCMAKE_BUILD_TYPE=Debug
            if is_debug:
                subprocess.run("cmake -DCMAKE_BUILD_TYPE=Debug -G \"Visual Studio 15 2017 Win64\" ../")
                subprocess.run("cmake --build . --config Debug")
            else:
                subprocess.run("cmake -DCMAKE_BUILD_TYPE=Release -G \"Visual Studio 15 2017 Win64\" ../")
                subprocess.run("cmake --build . --config Release")
            
        elif machine_type == 'x86':
            if is_debug:
                subprocess.run("cmake -DCMAKE_BUILD_TYPE=Debug -G \"Visual Studio 15 2017\" ../")
                subprocess.run("cmake --build . --config Debug")
            else:
                subprocess.run("cmake -DCMAKE_BUILD_TYPE=Release -G \"Visual Studio 15 2017\" ../")
                subprocess.run("cmake --build . --config Release")
        else:
            print('unsupport machine')
    elif system_type == 'Linux':
        # -DCMAKE_BUILD_TYPE=Debug
        if is_debug:
            subprocess.run("cmake -DCMAKE_BUILD_TYPE=Debug -G \"Unix Makefiles\" ../")
            subprocess.run("cmake --build . --config Debug")
        else:
            subprocess.run("cmake -DCMAKE_BUILD_TYPE=Release -G \"Unix Makefiles\" ../")
            subprocess.run("cmake --build . --config Release")
    else:
        print('unsupport system')
    
    

    os.chdir("../")
    
    if system_type == 'Windows':
        if is_debug:
            shutil.copy('build/traph/source/interface/traph_tensor.py', 'python/pytraph/core')
            shutil.copy('build/traph/source/interface/Debug/_traph_tensor.pyd', 'python/pytraph/core')
        else:
            shutil.copy('build/traph/source/interface/traph_tensor.py', 'python/pytraph/core')
            shutil.copy('build/traph/source/interface/Release/_traph_tensor.pyd', 'python/pytraph/core')
    elif system_type == 'Linux':
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