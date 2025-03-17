from moviepy.video.io.VideoFileClip import VideoFileClip


# python others/trimming_mp4.py
# INPATH = './data/mp4/61thSingleDance_1k.mp4'
INPATH = './data/mp4/kawadame_ebisulive.mp4'
START = 0
END = 10


def main():
    video = VideoFileClip(INPATH)
    trimmed_video = video.subclip(START, END)
    # OUTPATH = INPATH.replace('full_1k.mp4', f'{START}_{END}_1k.mp4')
    OUTPATH = INPATH.replace('.mp4', f'_{START}_{END}.mp4')
    trimmed_video.write_videofile(OUTPATH, codec="libx264", audio=True, audio_codec="aac")


if __name__ == '__main__':
    main()