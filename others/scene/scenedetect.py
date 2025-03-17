from scenedetect import detect, ContentDetector, split_video_ffmpeg


# python others/scenedetect.py
VIDEOPATH = './data/mp4/62thSingleMV_scene.mp4'


def split_video_into_scenes(threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(VIDEOPATH)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    split_video_ffmpeg(VIDEOPATH, scene_list, show_progress=True)


if __name__ == '__main__':
    split_video_into_scenes()