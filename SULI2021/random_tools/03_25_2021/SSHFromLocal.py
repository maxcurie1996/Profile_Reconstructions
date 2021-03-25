import sys
import os
from ..main import *

def main():

    print(os.getcwd())
    hello()

if __name__ == "__main__":
    try:
        if sys.argv[1] == 'deploy':
            import paramiko
            host = 'portal.pppl.gov'
            '''
            username = input('Enter Username: ')
            pwd = getpass('Enter Password: ')
            '''
            username = 'jzimmerm'
            pwd = '1969Mb429('
            ### REMOVE BEFORE SENDING

            ssh = paramiko.client.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, username=username, password=pwd)

            sftp = ssh.open_sftp()
            sftp.put(__file__, '/tmp/script.py')
            sftp.put(os.path.abspath('../main.py'), '/tmp/import_mat.py')
            sftp.close()

            stdin, stdout, stderr = ssh.exec_command('cd /tmp;python /tmp/script.py')
            for line in stderr:
                # Process each line in the remote output
                print(line)
            for line in stdout:
                # Process each line in the remote output
                print(line)

        ssh.close()
        sys.exit(0)
    except IndexError:
        pass

# No cmd-line args provided, run script normally
main()