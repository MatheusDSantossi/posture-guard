import winsound

def play_sound():
    try:
        winsound.PlaySound(
            r"posture_guard\ui\assets\audio\watch-your-posture-keep-those-glutes-tucked-in.wav",
            winsound.SND_FILENAME | winsound.SND_ASYNC,
        )
        print("Playing...")
    except Exception as e:
        print(e)
