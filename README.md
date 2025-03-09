# object-tracker

Object Tracking using Yolov8 and OpenCV, served using FastAPI.

## How to Use

First, run:

```
pip install -r requirements.txt
```

To run the code, simply run:

```
python main.py --url <your video directory/URL>
```

Untuk menggambar region, ada dua opsi:

1. Pertama tekan tombol p di keyboard dan klik kanan menggunakan mouse. Kemudian klik empat titik yang ingin dijadikan region, bila selesai bisa tekan tombol p lagi
2. Koordinat bisa diubah melalui 

```
curl -X 'POST'   'http://localhost:8000/api/config/area'   -H 'Content-Type: application/json'   -d '{
  "points": [
    [100, 100],
    [300, 100],
    [200, 300],
    [100, 300]
  ]
}'
```

## Database Design

Sesuai dengan Challenge 1, desain database untuk program ini adalah:

| First Header  | Second Header |
| ------------- | ------------- |
| track_id | int  |
| event  | varchar  |
| timestamp  | varchar  |
| coordinates  | varchar  |

data ini ditampilkan pada endpoint /api/stats/.

- track_id: ID yang diberikan pada setiap orang yang masuk ke dalam region yang digambar
- event: menentukan apakah orang tersebut masuk/keluar dari region
- timestamp: waktu saat orang dengan ID tertentu masuk/keluar
- coordinate: koordinat region yang sudah digambar.

Adapun juga program menampilkan jumlah total orang yang masuk dan keluar dari area region melalui live video feed.

## Pengumpulan Dataset

Video sumber diambil dari website https://cctv.jogjakota.go.id/home.

Salah satu live streaming spesifik yang digunakan adalah [NolKM Timur](https://cctvjss.jogjakota.go.id/malioboro/NolKm_Timur.stream/playlist.m3u8)

## Feature Checklist

[x] Desain Database: Sudah ditulis di Readme bagian Desain Database
[x] Pengumpulan Dataset: Sudah dijelaskan pada bagian Pengumpulan Dataset
[x] Object Detection & Tracking: Sudah dilakukan dengan menggunakan YOLO v8. Object tracking menggunakan bytetrack.
[x] Counting & Polygon Area: People counting sudah diimplementasi dan region sudah dibuat bisa diubah sesuai input user
[x] Integrasi API: Sudah dilakukan dengan FastAPI. Belum sempat integrasi menggunakan dashboard.
[x] Deployment: Sudah dilakukan, namun belum sempat dilakukan containerization.



