import pytube

url = input("URL: ")
instance = pytube.YouTube(url)
video_stream = instance.streams.get_highest_resolution()
video_stream.download()
print(instance)