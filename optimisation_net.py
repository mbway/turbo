'''
Networking utilities for the optimiser library
'''

import time
import json
import struct
import socket
import hashlib


# This is to prevent deadlock but if a socket times out then it may be fatal. At
# least this way the problem is visible and you can restore from a checkpoint
# rather than have it sit in deadlock forever.
LAST_RESORT_TIMEOUT = 25.0 # seconds


def encode_JSON(obj, encoder=None):
    ''' encode an object in a JSON format ready to be sent over the network '''
    return json.dumps(obj, cls=encoder).encode('utf-8')
def decode_JSON(data):
    ''' decode JSON data sent over the network to an object '''
    return json.loads(data.decode('utf-8'))

def empty_msg():
    ''' an empty payload. Primary used as an acknowledgement. '''
    return bytes()
def never_stop():
    ''' pass as the value for should_stop '''
    return False


def send_msg(conn, payload):
    '''
    send a message (bytes) down the given connection.
    Message format:
        [4:length | 4:length | 16:checksum | *:payload]
    conn: the connection to send through
    payload: bytes to send
    '''
    assert isinstance(payload, bytes)
    # ! => network byte order (Big Endian)
    # I => unsigned integer (4 bytes)
    length = struct.pack('!I', len(payload))
    checksum = hashlib.md5(payload).digest()
    assert len(checksum) == 16
    # a more wasteful but simpler alternative to an error-correcting code is to
    # just duplicate the length. This avoids the situation where the length is
    # corrupted and the receiver reads an incorrect amount (potentially waiting
    # indefinitely).
    conn.sendall(length + length + checksum + payload)

def recv_msg(conn):
    '''
    receive a message from the given connection
    raises ValueError if the message is malformed/corrupted
    '''
    # length (4 bytes), length (4 bytes), checksum (16 bytes)
    header = read_exactly(conn, 24)

    # the length should be duplicated
    length, length_check = struct.unpack('!II', header[:8])
    checksum = header[8:]

    if length != length_check:
        raise ValueError('message length corrupted: {}/{}'.format(length, length_check))

    payload = empty_msg() if length == 0 else read_exactly(conn, length)

    if checksum != hashlib.md5(payload).digest():
        raise ValueError('checksum failed')

    return payload


def read_exactly(conn, num_bytes):
    '''
    read exactly the given number of bytes from the connection
    returns None if the connection closes before the number of bytes is reached
    '''
    data = bytes()
    while len(data) < num_bytes: # until the length is fully read
        left = num_bytes - len(data)
        # documentation recommends a small power of 2 to give best results
        chunk = conn.recv(min(4096, left))
        if len(chunk) == 0: # connection broken: will never receive any data over conn again
            raise socket.error('connection broke in read_exactly')
        else:
            data += chunk
    assert len(data) == num_bytes
    return data

def message_client(addr, connect_timeout, request,
                   should_stop, on_error):
    '''
    connect to a message server and send it the request, then read the response
    and return it. Return None if should_stop becomes True.

    addr: a tuple of (host, port) to connect to
    connect_timeout: the time between checking should_stop() while trying to connect
    request: bytes to send to the server
    should_stop: () -> bool
        whether to abort the communication, if aborted after the connection is
        formed: an empty request is sent and no response is expected. If this
        function returns because should_stop() becomes True then it will return None.
    on_error: (Exception) -> ()
        called when there is a _non_critical_ error with communication which
        will cause the entire communication to restart. You may want to sleep
        for a short time for example.
    return: either the response from the server, or None if should_stop became True
    '''
    while True:
        # try to connect
        connected = False
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
        sock.settimeout(connect_timeout) # only while trying to connect
        while not should_stop():
            try:
                sock.connect(addr)
                connected = True
                # should never time out, but don't want deadlock
                sock.settimeout(LAST_RESORT_TIMEOUT)
                break
            except socket.timeout:
                continue # this is normal and intentional to keep checking should_stop
            except socket.error as e:
                on_error(e)
                continue

        # request/response
        try:
            if should_stop():
                if connected:
                    send_msg(sock, empty_msg())
                return None
            else:
                send_msg(sock, request)
                response = recv_msg(sock)

                # if the response is non-empty then the server will be wanting
                # confirmation that it was received.
                #
                # if the confirmation fails to send with an exception, then the
                # interaction will re-start. It is assumed that if send crashes
                # then the server will not have received the confirmation and so
                # this is OK.
                if response != empty_msg():
                    send_msg(sock, empty_msg())

                return response

        except (socket.error, ValueError, socket.timeout) as e:
            # socket.error: transmission error
            # ValueError: corrupted message
            # also treat a timeout as an error since it is a last-resort timeout
            on_error(e)
            continue # restart entire communication
        finally:
            sock.close()

def message_server(server_addr, connect_timeout,
                   should_stop, handle_request, on_success):
    '''
    server_addr: a tuple of (host, port) to bind to
    connect_timeout: timeout for accepting client connections, determines how
        often should_stop is checked
    should_stop: () -> bool
        return: whether the server should stop
    handle_request: (request : bytes) -> (response : bytes)
        called when a message is received from a client.
        return: the response to the message
    on_success: (request : bytes) -> (response : bytes) -> ()
        called when an interaction completes successfully.
        Not called if the initial request message was empty.
    '''
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
        # able to re-use host/port combo even if in use
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(server_addr)
        sock.listen(16) # maximum number of connections allowed to queue
        # timeout for accept, not inherited by the client connections
        sock.settimeout(connect_timeout)

        conn = None
        while not should_stop():
            try:
                conn, addr = sock.accept()
                conn.settimeout(LAST_RESORT_TIMEOUT)

                request = recv_msg(conn)
                if request == empty_msg():
                    # client sent an empty request, does not expect a response
                    continue
                response = handle_request(request)
                send_msg(conn, response)

                # if the response is non-empty, expect a confirmation reply
                if response != empty_msg():
                    confirm = recv_msg(conn)
                    if confirm != empty_msg():
                        raise ValueError('Invalid confirmation message: "{}"'.format(confirm))

                on_success(request, response)

            except (socket.error, ValueError, socket.timeout) as e:
                # socket.error: transmission error
                # ValueError: corrupted message
                # socket.timeout: accept or conn timed out

                # 104: 'connection reset by peer' means that the socket is
                # broken and calling shutdown would cause an OSError
                connection_reset = hasattr(e, 'errno') and e.errno == 104

                if conn is not None and connection_reset:
                    conn.close() # no shutdown, only close
                    conn = None

                continue # restart communication
            finally:
                if conn is not None:
                    conn.shutdown(socket.SHUT_RDWR)
                    conn.close()
                    # reset to None so that it is not closed again if accept times out
                    conn = None
    finally:
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()


