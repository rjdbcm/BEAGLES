#!/usr/bin/env bash
# Crops an input video of arbitrary resolution into 9 separate videos
# corresponding to a 3x3 grid of equal aspect ratio sections.

BASENAME="${FILENAME%.*}"

COPY=$BASENAME

RESOLUTION=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$1")

WIDTH=$(echo $RESOLUTION | cut -f1 -dx)
HEIGHT=$(echo $RESOLUTION | cut -f2 -dx)

FIRSTY=`expr $HEIGHT / 3`
SECONDY=`expr $HEIGHT - $FIRSTY`

FIRSTX=`expr $WIDTH / 3`
SECONDX=`expr $WIDTH - $FIRSTX`

echo $FIRSTY $SECONDY
EXTENSION=".""${1##*.}"

xs=(0 0 $FIRSTX $SECONDX 0 $FIRSTX $SECONDX 0 $FIRSTX $SECONDX)
ys=(0 0 0 0 $FIRSTY $FIRSTY $FIRSTY $SECONDY $SECONDY $SECONDY)


for i in $(seq 1 9); do

    BASENAME="$BASENAME""_camera""$i$EXTENSION"
    echo $BASENAME ${xs[i]} ${ys[i]}
    ffmpeg -i "$1" -filter:v "crop=$FIRSTX:$FIRSTY:${xs[i]}:${ys[i]}" -c:a copy $BASENAME
    BASENAME=$COPY

done

