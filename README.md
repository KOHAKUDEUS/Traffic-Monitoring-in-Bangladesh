# 🚀 Dự án Nhận diện Đối tượng Thời gian Thực (Real-time Object Detection) với YOLOv8

## ✨ Tổng quan Dự án
Dự án này giới thiệu một **pipeline hoàn chỉnh** cho việc xây dựng, huấn luyện và triển khai mô hình **Nhận diện Đối tượng Real-time** sử dụng kiến trúc **YOLOv8**. Quy trình tập trung vào việc **kiểm soát chặt chẽ dữ liệu (Data Governance)**, từ chuẩn hóa cấu trúc folder, tự động hóa chuyển đổi annotation, quản lý Metadata mở rộng, đến áp dụng các kỹ thuật **Data Augmentation** để tối ưu hóa hiệu suất mô hình và tăng khả năng tổng quát hóa.

---

## 🛠️ Công nghệ và Kiến trúc

| **Hạng mục**         | **Chi tiết** |
|-----------------------|-------------|
| **Kiến trúc Mô hình** | YOLOv8s (phiên bản *small*) từ thư viện **ultralytics**. |
| **Hiệu suất Real-time** | Tốc độ suy luận (Inference) được tối ưu hóa, đảm bảo vận hành với **độ trễ thấp**. <br> Thời gian suy luận trung bình mỗi ảnh: **~10–15 ms**. |
| **Thiết bị (Device)** | Sử dụng **GPU (CUDA)** để đảm bảo hiệu suất cao nhất. |

---

## 📂 Kiểm soát Folder và Chuẩn bị Dữ liệu

### 1. Thiết kế Cấu trúc Thư mục
Cấu trúc thư mục được tuân thủ **nghiêm ngặt theo chuẩn YOLO** để dễ dàng tích hợp với pipeline huấn luyện của **ultralytics**.

data_yolo/  
├── images/  
│   ├── train/  # Tập ảnh huấn luyện (Image(train))   
│   └── val/    # Tập ảnh kiểm tra/đánh giá (Image(val))  
└── labels/  
├── train/  # Tập tin annotation YOLO (.txt) (labels(train))  
└── val/    # Tập tin annotation YOLO (.txt) (labels(val))  

> Các thư mục này được tạo ra để lưu trữ ảnh và nhãn đã chuyển đổi, đảm bảo **phân chia rõ ràng** giữa tập huấn luyện và tập kiểm tra.

---

### 2. Chuyển đổi Annotation (XML Parsing)
Dữ liệu annotation ban đầu (định dạng **XML Pascal VOC**) đã được xử lý để trích xuất thông tin và chuyển đổi sang **định dạng YOLO**.

#### Trích xuất thông tin:
- Sử dụng thư viện `xml.etree.ElementTree` để parse file XML.
- Trích xuất:
  - `filename`
  - `size` (width, height)
  - `object` (name - tên lớp)
  - `bndbox` (xmin, ymin, xmax, ymax - tọa độ pixel)

#### Chuyển đổi định dạng:
- **Đầu vào**: `[xmin, ymin, xmax, ymax]` (pixel)
- **Đầu ra (YOLO format)**:  
  `class_index x_center y_center width height`  
  *(tọa độ chuẩn hóa - normalized)*

#### Danh sách lớp (Classes):
- Mã hóa **21 lớp đối tượng giao thông** thành chỉ mục số (0 đến 20)  
  Ví dụ: `'car': 5`, `'bus': 4`, `'rickshaw': 13`

---

### 3. Thiết kế Metadata và Ghi vào Disk
Một tệp **metadata** (`metadata.yaml` hoặc `metadata.csv`) đã được thiết kế để lưu trữ **thông tin chi tiết (rất nhiều)** về tập dữ liệu, giúp:
- Quản lý hiệu quả
- Phân tích chất lượng dữ liệu

---

## 🔄 Tăng cường Dữ liệu (Data Augmentation)
Để tăng cường **tính bền vững** và **khả năng tổng quát hóa** của mô hình, tránh **overfitting**, **Data Augmentation** được thực hiện bằng cách sử dụng các phép biến đổi ảnh và **bounding box đồng bộ**.

| **Loại biến đổi**     | **Kỹ thuật được áp dụng (Ví dụ)**                     | **Thư viện**                  |
|-----------------------|-------------------------------------------------------|-------------------------------|
| **Biến đổi Hình học** | `RandomPerspective`, `RandomFlip` (Lật ngang/dọc)     | `torchvision.transforms.v2`   |
| **Biến đổi Màu sắc**  | `ColorJitter` (Brightness, Contrast, Saturation, Hue) | `torchvision.transforms.v2`   |
| **Kỹ thuật tổng hợp** | `Mosaic`, `MixUp` (Tích hợp sẵn trong YOLOv8)         | `ultralytics`                 |

---
