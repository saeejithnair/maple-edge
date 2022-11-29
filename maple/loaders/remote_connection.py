"""This script uses the Fabric library to connect to a remote server and
execute a command. It is used to transfer models between the host and
target device, as well as store the results of latency profiling on the
host device."""

from fabric import Connection
from patchwork.files import exists


class RemoteConnection:
    def __init__(self, user, ip):
        self.user = user
        self.ip = ip

        self._connect()

    def __del__(self):
        """Closes the connection to the remote device upon destruction."""
        try:
            self.connection.close()
        except AttributeError:
            pass

    def _connect(self):
        """Creates and opens a connection to the remote device."""
        self.connection = Connection(
            user=self.user, host=self.ip)

        self.connection.open()

    def close(self):
        """Closes the connection to the remote device."""
        self.connection.close()

    def check_exists(self, remote_path):
        return exists(self.connection, remote_path)

    def put(self, local_path, remote_path, num_retries=3):
        """Transfers a file from the local device to the remote device."""
        return self.connection.put(local=local_path, remote=remote_path,
                                   preserve_mode=True)

    def get(self, remote_path, local_path, num_retries=3):
        """Transfers a file from the remote device to the local device."""
        return self.connection.get(remote=remote_path, local=local_path,
                                   preserve_mode=True)
