#! /bin/sh

NAME="RAMDisk"

function RAMDisk_mount() {
    if [[ ! -d "Volumes/$NAME" ]]; then
        diskutil eraseVolume HFS+ $NAME `hdiutil attach -nomount ram://$((2048 * 2))`
    fi
}

function RAMDisk_unmount() {
    while true
    do
        CURDISK=$(diskutil info RAMDisk | grep -o '/dev/disk[1-99]')
        echo $1 $CURDISK
        if [[ "$CURDISK" = "" ]]
        then
            exit
        else
            hdiutil detach $CURDISK
        fi
    done
}

eval ${NAME}_${1}
#if [[ "$1" = "mount" ]]; then
#    RAMDisk_mount
#elif [[ "$1" = "unmount" ]]; then
#    RAMDisk_unmount
#else
#    >&2 echo "line $LINENO: $NAME_$1: command not found"
#fi