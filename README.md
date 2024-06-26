# SSC_visual_effect

Tham số:
```
  --audio  Đường dẫn file audio, định dạng mp3.
  --layer1  Đường dẫn file hình ảnh background.
  --mode  Loại dữ liệu RMS, hình ảnh sẽ nhảy theo dạng sóng. Gồm ['bass', 'treble', 'mid', 'hpss']
  --range  Độ chuyển của hiệu ứng zoom. Khoảng (từ 20, đến 120) 
  --domain Đường dẫn Authen
  --token  token authen
  --threads Số luồng chạy trên CPU, mặc định là 3 luồng
  --output Đường dẫn file ouput, mặc định là output_video.mp4
  --effect Hiệu ứng hình ảnh theo sóng nhạc. Gồm ['zoom', 'blur', 'brightness', 'rgb']
```

# Sample command
### Render with image input

```
python Reactive_music.py --audio "../resource/Sac_Moi_Em_Hong.mp3" --layer1 "../resource/nhac_de_yeu.jpg" --mode bass --range 30 --domain "http://evision-api.sspartner.co/api/v2/userinfo" --token "6329|qeBohgvUP5PRwdqkZluuNQobPQJILh0s9sfyUyYV" --output "../output_image.mp4" --threads 3 --effect zoom
```

```
Reactive_music.exe --audio "../../../resource/Sac_Moi_Em_Hong.mp3" --layer1 "../../../resource/nhac_de_yeu.jpg" --mode bass --range 30 --domain "http://evision-api.sspartner.co/api/v2/userinfo" --token "6329|qeBohgvUP5PRwdqkZluuNQobPQJILh0s9sfyUyYV" --output "../../../output_image.mp4" --effect zoom --threads 3
```

### Render with video input
```
python Reactive_music.py --audio "../resource/output_circular_bars_audio.mp3" --layer1 "../resource/output_circular_bars.mp4" --mode bass --range 30 --domain "http://evision-api.sspartner.co/api/v2/userinfo" --token "6329|qeBohgvUP5PRwdqkZluuNQobPQJILh0s9sfyUyYV" --output "../output_image.mp4" --effect zoom --threads 3
```

```
Reactive_music.exe --audio "../../../resource/output_circular_bars_audio.mp3" --layer1 "../../../resource/output_circular_bars.mp4" --mode bass --range 30 --domain "http://evision-api.sspartner.co/api/v2/userinfo" --token "6329|qeBohgvUP5PRwdqkZluuNQobPQJILh0s9sfyUyYV" --output "../../../output_image.mp4" --effect zoom --threads 3
```

# Build Application with PyInstaller

```
pyinstaller --icon=favicon.ico Reactive_music.py
```
