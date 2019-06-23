#! /bin/sh
VOLUMEPATH="/Volumes"
NAME="RAMDisk"

function RAMDisk_mount() {
    if [[ ! -d "$VOLUMEPATH/$NAME" ]]; then
        diskutil eraseVolume HFS+ $NAME `hdiutil attach -nomount ram://$((2048 * 2))`
    else
        >&2 echo Error: $NAME already mounted in /Volumes
        exit 1
    fi
}

function RAMDisk_unmount() {
    while true
    do
        CURDISK=$(diskutil info RAMDisk | grep -o '/dev/disk[1-99]')
        if [[ "$CURDISK" = "" ]]
        then
            exit
        else
            echo "$1 $NAME @ $CURDISK"
            hdiutil detach $CURDISK
        fi
    done
}

eval ${NAME}_${1} ${1}
#if [[ "$1" = "mount" ]]; then
#    RAMDisk_mount
#elif [[ "$1" = "unmount" ]]; then
#    RAMDisk_unmount
#else
#    >&2 echo "line $LINENO: $NAME_$1: command not found"
#fi