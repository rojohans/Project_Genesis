import os


os.system("ffmpeg -f image2 -s 1000x1000  -r 20 -i anim%03d.png -vcodec mpeg4 -b 6M -y Test_Video.mp4")
#os.system("ffmpeg -r 10 -i anim%03d.png -vcodec mpeg4 -y movie.mp4")


