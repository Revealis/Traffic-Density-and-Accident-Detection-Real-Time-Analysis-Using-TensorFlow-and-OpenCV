import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Yerel dizinden modeli yükleyin
model_dir = "ssd_mobilenet_v2"
model = tf.saved_model.load(model_dir)

# Video dosyasını yükle
cap = cv2.VideoCapture('MOT17-14-DPM-raw.webm')

# Araç ve yaya sayıları
vehicle_count = 0
pedestrian_count = 0

# Trafik yoğunluğu verilerini saklamak için liste
traffic_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü modelin kabul edeceği formata çevir
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Modeli çalıştır ve tahminleri al
    detections = model(input_tensor)

    # Tahminleri yorumlama
    num_detections = int(detections.pop('num_detections'))
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int64)
    detection_scores = detections['detection_scores'][0].numpy()
    detection_boxes = detections['detection_boxes'][0].numpy()

    # Tespit edilen nesneleri çiz ve say
    current_vehicle_count = 0
    current_pedestrian_count = 0
    for i in range(num_detections):
        if detection_scores[i] > 0.5:  # Eşik değeri
            y_min, x_min, y_max, x_max = detection_boxes[i]
            (start_x, start_y, end_x, end_y) = (int(x_min * frame.shape[1]), int(y_min * frame.shape[0]),
                                                int(x_max * frame.shape[1]), int(y_max * frame.shape[0]))
            label = detection_classes[i]

            if label in [2, 3, 4, 6, 8]:  # Araç sınıfları (örneğin 2: bicycle, 3: car, 4: motorcycle, 6: bus, 8: truck)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                current_vehicle_count += 1
            elif label == 1:  # Yaya sınıfı
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
                current_pedestrian_count += 1

    vehicle_count += current_vehicle_count
    pedestrian_count += current_pedestrian_count

    # Mevcut tespit verilerini kaydet
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    traffic_data.append([timestamp, current_vehicle_count, current_pedestrian_count])

    # İşlenmiş görüntüyü göster
    cv2.imshow('Traffic Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Trafik yoğunluğu analizini görselleştir
counts = {'Vehicles': vehicle_count, 'Pedestrians': pedestrian_count}

sns.barplot(x=list(counts.keys()), y=list(counts.values()))
plt.title('Total Traffic Density Analysis')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# Toplanan trafik verilerini DataFrame'e dönüştür
df = pd.DataFrame(traffic_data, columns=['Timestamp', 'Vehicles', 'Pedestrians'])

# Zaman bazlı analiz ve görselleştirme
plt.figure(figsize=(12, 6))
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
df['Vehicles'].plot(label='Vehicles', color='b')
df['Pedestrians'].plot(label='Pedestrians', color='r')
plt.title('Traffic Density Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.show()

# Trafik kazası tespiti (anormal hareket)
df['Vehicle_Delta'] = df['Vehicles'].diff().fillna(0)
df['Pedestrian_Delta'] = df['Pedestrians'].diff().fillna(0)


# Anormal artışları tespit et (örnek eşik değerlerini değiştirerek)
anomalies = df[(df['Vehicle_Delta'] > df['Vehicle_Delta'].mean() + 2*df['Vehicle_Delta'].std()) |
               (df['Pedestrian_Delta'] > df['Pedestrian_Delta'].mean() + 2*df['Pedestrian_Delta'].std())]


# Anomali sonuçlarını yazdır
print("Potential Accident Anomalies:")
print(anomalies)

# Basit Trafik Yoğunluğu Raporu
print("\nBASİT TRAFİK YOĞUNLUĞU RAPORU\n")
for index, row in anomalies.iterrows():
    print(f"{index} - {row['Pedestrian_Delta']} pedestrian increase and {row['Vehicle_Delta']} vehicle increase was detected")

print(f"\nTotal number of vehicles detected: {vehicle_count}")
print(f"Total number of pedestrians detected: {pedestrian_count}")


# Calculate the accident probability based on the rate of anomalies
anomaly_count = len(anomalies)
total_time_intervals = len(df)
accident_probability = (anomaly_count / total_time_intervals) * 100

print(f"Estimated Accident Probability: {accident_probability:.2f}%")


# Raporlama ve analiz
df.to_csv('traffic_analysis_report.csv')
print("Traffic analysis report saved as 'traffic_analysis_report.csv'")
