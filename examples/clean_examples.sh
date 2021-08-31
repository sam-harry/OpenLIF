#!/bin/sh

find . -type f \( -name "*.mp4" -o -name "*.png" -o -name "*.out" -o -name "Thumbs.db" \) -delete -print
find . -type d -name "overlay_lif" -delete -print
