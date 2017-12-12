#!/bin/bash

# simulate a bad connection on the loopback interface for testing purposes.
#
# also check out `libfiu` (packages available in Ubuntu) which can simulate
# errors by injecting them into function calls themselves rather than
# interfering with the connection.
#
# examples:
# fiu-run -x -c 'enable name=posix/io/net/connect' python3 my_program.py
# fiu-run -x -c 'enable_random name=posix/io/net/send,probability=0.5' python3 my_program.py
#
# the `name=` is the name of the function to sabotage by causing one of the
# possible errors. See the posix/io/net section of:
# https://github.com/albertito/libfiu/blob/master/preload/posix/modules/posix.io.mod

DEV=lo

#### tc terminology explanation ####
# tc = traffic control
# qdisc = queueing discipline
# root = rules are nested in a tree so root inserts them at the top
# netem = net emulation


shutdown() {
    echo "removing simulation"
    tc qdisc del dev $DEV root netem
    ip link set $DEV up
    exit
}
# shutdown gracefully on Ctrl+C
trap shutdown SIGINT

# ensure root
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root"
   exit 1
fi

# add net emulation to the device
tc qdisc add dev $DEV root netem

echo "simulating a bad connection"

# base delay with some variation.
# delay TIME JITTER CORRELATION
# loss a b => probability of a unless previous packet dropped, then probability of b
tc qdisc change dev $DEV root netem      \
    delay 50ms 50ms distribution normal  \
    loss 20% 25%                         \
    duplicate 5%                         \
    corrupt 5%                           \
    reorder 10%


# unplug the interface for the given length of time
unplug_interface() {
    ip link set $DEV down
    echo "interface down"
    sleep $1
    ip link set $DEV up
    echo "interface up"
}


# repeatedly pull the plug on the interface
while true; do
    unplug_interface 10
    sleep 20
done
