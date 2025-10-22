

# 🛣️ US Road Signs Detection Model
*YOLOv8-based road sign detection trained on Roboflow dataset*

📅 **Updated:** Oct 17, 2025  
📦 **Dataset:** [US Road Signs (Roboflow)](https://universe.roboflow.com/signage/us-road-signs-9xtlj)  
🧾 **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## 🚀 Overview
This project uses **YOLOv8** to detect and classify **U.S. road signs** in images and videos.  
It helps identify common signs like *Stop*, *Speed Limit*, *Yield*, and more — enabling real-time traffic and safety analysis.

---

## 💡 Use Cases
- 🚗 **Driver Assistance:** Recognize signs for autonomous or assisted driving.  
- 🛠️ **Road Maintenance:** Detect damaged or missing signs for inspection.  
- 🗺️ **Mapping & Navigation:** Improve road sign accuracy in map data.  
- 🎓 **Driver Education:** Help new drivers learn sign meanings.  
- 🚦 **Traffic Analysis:** Monitor sign compliance using CCTV or drone footage.

)
model.train(data='data.yaml', epochs=50, imgsz=640, batch=8, name='US_Road_Signs_Model', device='cpu')
