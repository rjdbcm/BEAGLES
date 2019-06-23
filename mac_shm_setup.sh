#! /bin/sh

NAME="RAMDisk"

function RAMDisk_mount() {
    diskutil quiet eraseVolume HFS+ $NAME `hdiutil attach -nomount ram://$((2048 * 2))`
}

function RAMDisk_unmount() {
    while true
    do
        CURDISK=$(diskutil info RAMDisk | grep -o '/dev/disk[1-99]')
        echo ${1} $CURDISK
        if [[ "$CURDISK" = "" ]]
        then
            exit
        else
            hdiutil detach -quiet $CURDISK
        fi
    done
}

eval ${NAME}_${1}
#if [[ "$1" = "mount" ]]; then
#    RAMDisk_mount
#fi
#
#if [[ "$1" = "unmount" ]]; then
#    RAMDisk_unmount
#fi