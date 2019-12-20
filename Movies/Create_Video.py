import os
import System_Info
#
# There are slight differences in the commands to be used on windows/mac OS.
# windows : "ffmpeg ..."
#     mac : "./ffmpeg ..."
#


if System_Info.OPERATING_SYSTEM is 'MAC_OS':
    os.system("./ffmpeg -f image2 -s 700x700  -r 20 -i anim%03d.png -vcodec mpeg4 -b 6M -y Test_Video.mp4")
else:
    os.system("ffmpeg -f image2 -s 1000x1000  -r 20 -i anim%03d.png -vcodec mpeg4 -b 6M -y Test_Video.mp4")
#os.system("ffmpeg -r 10 -i anim%03d.png -vcodec mpeg4 -y movie.mp4")


