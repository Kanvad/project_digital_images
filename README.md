# ĐỒ ÁN CUỐI KỲ: NHẬP MÔN XỬ LÝ ẢNH SỐ 
**Đề tài:** Ứng dụng xử lý ảnh để đo kích thước vật thể trong ảnh (đo chiều dài, diện tích)

### CamRuler - Công cụ đo lường và đếm đối tượng dựa trên camera

Chương trình là một tập lệnh Python sử dụng OpenCV để ghi lại video từ camera và đo kích thước của các đối tượng trong luồng video. Nó cũng đếm số lượng đối tượng trong khung hình video.

# Các tính năng chính

* Quay video từ camera
* Đo kích thước của các đối tượng trong luồng video
* Đếm số lượng đối tượng trong khung hình video
* Lưu lịch sử đo lường vào tệp
* Xuất lịch sử đo lường ra tệp CSV
* Xoay khung hình video 180 độ
* Thay đổi đơn vị đo lường (cm, inch, mm, v.v.)

## Cách sử dụng

1. Cài đặt các gói cần thiết bằng pip: `pip install -r requirements.txt`
2. Chạy tập lệnh bằng Python: `python camruler.py`
3. Chọn camera cần sử dụng (mặc định là camera 0)
4. Di chuyển camera để định vị đối tượng trong khung hình
5. Nhấn phím cách để quay đối tượng và bắt đầu đo
6. Nhấn phím 'c' để vào chế độ cấu hình (config)
7. Nhấn phím 'q' để thoát chương trình

## Chế độ cấu hình

Trong chế độ cấu hình, bạn có thể thay đổi các cài đặt sau:

* Đơn vị đo lường (cm, inch, mm, v.v.)
* Kích thước đối tượng tối thiểu (phần trăm khung hình)
* Giá trị ngưỡng để phát hiện đối tượng
* Giá trị làm mờ để phát hiện đối tượng

Nhấn phím 'c' để thoát chế độ cấu hình.

## Lịch sử đo lường

Lịch sử đo lường được lưu vào tệp có tên `camruler_history.json`. Bạn có thể xuất lịch sử đo lường sang tệp CSV bằng cách nhấn phím 'e'.

## Đếm đối tượng

Tính năng đếm đối tượng được bật theo mặc định. Bạn có thể tắt tính năng này bằng cách nhấn phím 'o'.

## Phân loại hình dạng

Tính năng phân loại hình dạng được tắt theo mặc định. Bạn có thể bật tính năng này bằng cách nhấn phím 's'.

## Xoay khung hình video

Bạn có thể xoay khung hình video 180 độ bằng cách nhấn phím 'r'.

## Thay đổi đơn vị đo lường

Bạn có thể thay đổi đơn vị đo lường bằng cách nhấn phím 'u'.

## Phần trăm tối thiểu

Bạn có thể thay đổi kích thước đối tượng tối thiểu (phần trăm khung hình) bằng cách nhấn phím 'p'.

## Ngưỡng

Bạn có thể thay đổi giá trị ngưỡng để phát hiện đối tượng bằng cách nhấn phím 't'.

## Làm mờ

Bạn có thể thay đổi giá trị làm mờ để phát hiện đối tượng bằng cách nhấn phím 'b'.