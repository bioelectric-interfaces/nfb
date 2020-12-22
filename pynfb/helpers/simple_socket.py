# Echo server program
import socket
from time import sleep
import numpy as np

HOST = ''  # Symbolic name meaning all available interfaces
PORT = 50007  # Arbitrary non-privileged port
META_STR_LENGTH = 3 # length of meta info in the message


class EchoError(BaseException):
    pass

class SimpleSocket:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    @staticmethod
    def encode(meta_str, obj):
        if len(meta_str) != META_STR_LENGTH:
            raise ValueError('Meta info string should be lower then {}'.format(META_STR_LENGTH))
        if isinstance(obj, np.ndarray):
            return b'n' + meta_str.encode() + obj.tobytes()
        if isinstance(obj, str):
            return b's' + meta_str.encode() + obj.encode()
        raise TypeError('Cannot encode type: {}'.format(type(obj)))

    @staticmethod
    def decode(buffer):
        type_str = buffer[:1].decode()
        meta_str = buffer[1:1+META_STR_LENGTH].decode()
        obj_buffer = buffer[1+META_STR_LENGTH:]
        if type_str == 'n':
            obj = np.frombuffer(obj_buffer, float)
        elif type_str == 's':
            obj = obj_buffer.decode()
        else:
            raise ValueError('Cannot read type: {}'.format(type_str))
        return meta_str, obj

    def close(self):
        self.s.close()


class SimpleServer(SimpleSocket):
    def __init__(self):
        super(SimpleServer, self).__init__()
        self.s.bind((HOST, PORT))
        self.s.listen(1)
        print('Waiting clients')
        self.conn, addr = self.s.accept()
        self.conn.setblocking(False)
        print('Connected by', addr)

    def pull_message(self):
        received_data = (None, None)
        try:
            buffer = self.conn.recv(1024)
            if len(buffer):
                received_data = self.decode(buffer)
                self.conn.sendall(self.encode(*received_data))
        except BlockingIOError:
            pass
        return received_data

    def close(self):
        if self.conn is not None:
            self.conn.close()
        super(SimpleServer, self).close()


class SimpleClient(SimpleSocket):
    def __init__(self):
        super(SimpleClient, self).__init__()
        self.s.connect((HOST, PORT))
        print('Connected to', (HOST, PORT))

    def push_message(self, meta_str, obj):
        buffer = self.encode(meta_str, obj)
        self.s.sendall(buffer)
        buffer2 = self.s.recv(1024)
        meta_str2, obj2 = self.decode(buffer2)
        if isinstance(obj2, np.ndarray):
            if not np.allclose(np.frombuffer(obj2, float), obj):
                raise EchoError("Received array didn't match sent")
        elif isinstance(obj2, str):
            if not obj==obj2:
                raise EchoError("Received string didn't match sent")
        else:
            raise EchoError("Received type not recognized")


if __name__ == '__main__':
    # client side
    from threading import Thread
    def client_sim():
        c = SimpleClient()
        for k in range(100):
            if k % 2:
                r = np.random.randn(32 + 2 + 1)
                c.push_message('chs', r)
            else:
                r = 'MMFF{}'.format(k)
                c.push_message('msg', r)
            print('Client >> Sent', r)
            sleep(1 / 10)
        c.close()
        print('Client >> Done.')
    client_thread = Thread(target=client_sim)
    client_thread.start()

    # server side
    server = SimpleServer()
    server.accept()
    while 1:
        meta_str, obj = server.pull_array()
        if obj is not None:
            print('Received data:', obj, meta_str)
