# Check wich computer to decide where the things are mounted
import socket
import sys
import os

comp_name = socket.gethostname()
print 'Computer: ' + comp_name

if 'txori' in comp_name or 'passaro' in comp_name or 'lintu' in comp_name or 'niao' in comp_name:
    repos_folder = os.path.abspath('/mnt/cube/earneodo/repos')

elif 'niao' in comp_name:
    kilo_tmp_folder = os.path.join('/home/earneodo/kilotmp')

sys.path.append(os.path.join(repos_folder, 'ephysflow'))
sys.path.append(os.path.join(repos_folder, 'analysis-tools'))

