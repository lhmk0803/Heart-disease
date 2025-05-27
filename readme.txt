Source bao gồm các file:
- data: chứa dataset ban đầu về bệnh tim và dataset sau khi tăng cường dữ liệu.
- jupyter: chứa source_code.ipynb
- model: chứa các model được tối ưu hóa.
- tempplates: chứa source giao diện input và result.
- app.py: định nghĩa ứng dụng web sử dụng Flask.

Hướng dẫn chạy dự án:
Mở file app.py trong thư mục Source để trực tiếp thực hiện việc dự đoán bệnh tim thông qua các mô hình đã được lưu ở thư mục "model"

Nếu bắt đầu từ đầu, thực hiện các bước sau:
1. Mở thư mục "jupyter", chạy file source_code.ipynb để thực hiện việc sử dụng và lưu các mô hình vào thư mục "model"
2. Thực hiện việc dự đoán thông qua file app.py ở thư mục "Source"