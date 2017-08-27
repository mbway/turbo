'''
Networking utilities for the optimiser library
'''

import time
import json
import struct
import socket


# set some timeout to prevent deadlock, but ideally should never timeout
LAST_RESORT_TIMEOUT = 25.0 # seconds

def send_json(conn, obj, encoder=None):
    '''
    send the given object through the given connection by first serialising the
    object to JSON.
    conn: the connection to send through
    obj: the object to send (must be JSON serialisable)
    encoder: a JSONEncoder to use
    '''
    data = json.dumps(obj, cls=encoder).encode('utf-8')
    # ! => network byte order (Big Endian)
    # I => unsigned integer (4 bytes)
    length = struct.pack('!I', len(data))
    conn.sendall(length + data)

def send_empty(conn):
    ''' send a 4 byte length of 0 to signify 'no data' '''
    conn.sendall(struct.pack('!I', 0))

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

def recv_json(conn):
    '''
    receive a JSON object from the given connection
    '''
    # read the length. Be lenient with the connection here since once the length
    # is received the peer obviously wants to communicate, but until then we are
    # not sure. If the connection breaks or times out before the length is read,
    # treat that as if a length of 0 was transmitted.
    data = read_exactly(conn, 4)
    length, = struct.unpack('!I', data) # see send_json for the protocol
    if length == 0:
        return None # indicates 'no data'
    else:
        data = read_exactly(conn, length)
        obj = json.loads(data.decode('utf-8'))
        return obj

def request_response_client(addr, timeout, should_stop, request,
                            error_wait, JSON_encoder=None):
    '''
    create a client socket and send a request down it, then wait for a response.
    timeout: the time between checking should_stop() while trying to connect
    should_stop: a function to query whether to abort the send and receive, if
        aborted after the connection is formed: an empty request is send and no
        response is expected. If this function returns because should_stop()
        becomes True then it will return 'should_stop'.
    request: an object to send as JSON
    '''
    while True:
        # try to connect
        connected = False
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
        sock.settimeout(timeout) # only while trying to connect
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
                time.sleep(error_wait)
                continue

        # request/response
        try:
            if should_stop():
                if connected:
                    send_empty(sock)
                return 'should_stop'
            else:
                send_json(sock, request, JSON_encoder)
                response = recv_json(sock)
                return response
        except socket.error as e:
            # transmission error anywhere => restart the whole thing
            time.sleep(error_wait)
            # TODO: shutdown or no? server uses shutdown if it has an error
            continue
        finally:
            sock.close()

def request_response_server(server_addr, timeout, handle_request,
                            should_stop, on_success, JSON_encoder=None):
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
        # able to re-use host/port combo even if in use
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(server_addr)
        sock.listen(16) # maximum number of connections allowed to queue
        sock.settimeout(timeout) # timeout for accept, not inherited by the client connections

        conn = None
        while not should_stop():
            try:
                conn, addr = sock.accept()
                conn.settimeout(LAST_RESORT_TIMEOUT)
                request = recv_json(conn)
                if request is None:
                    # client sent an empty request, does not expect a response
                    conn.shutdown(sock.SHUT_RDWR)
                    continue
                response = handle_request(request)
                if response is None:
                    send_empty(conn)
                else:
                    send_json(conn, response, encoder=JSON_encoder)

                on_success(request, response)
            except socket.error as e:
                if conn is not None:
                    conn.shutdown(socket.SHUT_RDWR)
                continue
            except socket.timeout: # for the accept or the client connection
                if conn is not None:
                    conn.shutdown(socket.SHUT_RDWR)
            finally:
                if conn is not None:
                    #TODO: move shutdown here but run tests again. The only case where shutdown is not called is the success case
                    conn.close()
                    # reset to None so that it is not closed again if accept times out
                    conn = None
    finally:
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()


